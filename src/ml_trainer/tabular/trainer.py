# type: ignore
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score
from sklearn.model_selection import KFold, _BaseKFold

from src.ml_trainer.models.base import EstimatorBase
from src.ml_trainer.tabular.evaluation.classification import macro_roc_auc_score
from src.ml_trainer.tabular.evaluation.regression import root_mean_squared_error
from src.ml_trainer.tabular.types import XyArrayLike
import json
from rich.console import Console
import joblib

console = Console()
REGRESSION_METRICS = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
}

BINARY_METRICS = {"auc": roc_auc_score}
MULTICLASS_METRICS = {"macro_auc": macro_roc_auc_score}


class Trainer:
    """Single Task Trainer for Tabular Data."""

    def __init__(
        self,
        estimators: list[EstimatorBase],  # lightgbm, etc
        ensemble: bool = False,
        out_dir: str | Path | None = None,
        # cross_validation params
        split_type: _BaseKFold | str | list[int] = KFold,  # if "fold" then use "fold" columns in the X (DataFrame)
        n_splits: int = 5,
        groups: ArrayLike | None = None,
        seed: int | None = None,
        eval_metrics: list[str] | None | str = "auto",
        custom_eval: callable | None = None,
        **kwargs,
    ) -> None:
        self.estimators = estimators
        self.ensemble = ensemble

        self.out_dir = Path(out_dir) or Path.cwd() / "output"
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.split_type = split_type
        self.n_splits = n_splits
        self.groups = groups
        self.seed = seed

        self.kwargs = kwargs
        self.is_fitted = False

    def get_eval_metrics(self, task: str) -> dict[str, callable]:
        if self.eval_metrics is None:
            if self.custom_eval is None:
                raise ValueError("eval_metrics or custom_eval must be specified")
            return {self.custom_eval.__name__: self.custom_eval}

        metrics = {}
        if task == "regression":
            metrics = REGRESSION_METRICS
        elif task == "binary":
            metrics = BINARY_METRICS
        elif task == "multiclass":
            metrics = MULTICLASS_METRICS
        else:
            raise ValueError("Invalid task")

        if isinstance(self.eval_metrics, list):
            metrics = {key: metrics[key] for key in self.eval_metrics}
            raise ValueError("Invalid eval_metrics")

        if self.custom_eval is not None:
            metrics[self.custom_eval.__name__] = self.custom_eval

        return metrics

    def train(
        self,
        X_train: XyArrayLike,
        y_train: XyArrayLike,
        X_val: XyArrayLike,
        y_val: XyArrayLike,
        out_dir: Path | None = None,
    ) -> None:
        if out_dir is not None:
            out_dir = self.out_dir

        out_dir.mkdir(exist_ok=True, parents=True)

        task = self.judge_task(y_train)
        metrics = self.get_eval_metrics(task)

        console.print(f"Estimator saving to: {out_dir}", style="bold green")
        console.print(f"[{task} Metrics]: \n{metrics}", style="bold green")

        resutls: dict = {}
        for estimator in self.estimators:
            estimator_uid = estimator.uid
            estimator_path = out_dir / estimator_uid / f"{estimator_uid}.pkl"

            console.print(f"[{estimator_uid}] start training :rocket:", style="bold blue")
            estimator.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            estimator.save(estimator_path)

            pred = estimator.predict(X_val)
            joblib.dump(pred, out_dir / estimator_uid / "pred.pkl")

            scores = {metric_name: metric(y_true=y_val, y_pred=pred) for metric_name, metric in metrics.items()}
            json.dump(scores, open(out_dir / estimator_uid / "scores.json", "w"), indent=4)

            console.print(f"[{estimator_uid}] scores: \n{json.dumps(scores, indent=4)}", style="bold blue")
            resutls[estimator_uid] = {
                "estimator": estimator,
                "pred": pred,
                "scores": scores,
            }

        self.is_fitted = True
        return resutls

    def cv(self, X_train: XyArrayLike, y_train: XyArrayLike):
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pl.DataFrame):
            if self.split_type == "fold" and "fold" in X_train.columns:
                folds = self._generate_folds(X_train["fold"])
                self.n_splits = X_train["fold"].nunique()
        elif isinstance(self.split_type, list):
            folds = self._generate_folds(self.split_type)
            self.n_splits = len(np.unique(self.split_type))
        else:
            folds = self.split_type(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            ).split(X_train, y_train, groups=self.groups)

        fold_fitted_results = {}
        for i_fold, (tr_idx, va_idx) in enumerate(folds):
            # dataframe と array に対応
            val_mask = np.zeros(len(X_train), dtype=bool)
            val_mask[va_idx] = True

            X_tr, X_va = X_train[tr_idx].copy(), X_train[va_idx].copy()
            y_tr, y_va = y_train[tr_idx].copy(), y_train[va_idx].copy()

            fitted_resuts = self.train(X_tr, y_tr, X_va, y_va, out_dir=self.out_dir / f"fold{i_fold}")
            fold_fitted_results[f"fold{i_fold}"] = fitted_resuts

        return fold_fitted_results

    def eval_oof(self, cv_results: dict):
        oof_results = {}  # {est1: pred, est2: pred, ...}

        for fold in cv_results:
            for est, data in cv_results[fold].items():
                scores = data["scores"]
                console.print(
                    f"[fold{fold}] [{est}] scores: \n{json.dumps(scores, indent=4)}",
                    style="bold blue",
                )
                if est not in oof_results:
                    oof_results[est] = data["pred"]
                else:
                    oof_results[est].extend(data["pred"])

        pass

    def predict(self):
        pass

    def predict_ensemble(self):
        pass

    def get_feature_importance(self):
        pass

    def plot_feature_importance(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def _generate_folds(self, fold_list: list | pd.Series):
        fold_series = pd.Series(fold_list)
        for fold in fold_series.unique():
            test_idx = fold_series[fold_series == fold].index
            train_idx = fold_series[fold_series != fold].index
            yield train_idx, test_idx

    def judge_task(self, y: ArrayLike) -> str:
        if len(np.unique(y)) == 2:
            return "binary"
        elif len(np.unique(y)) > 2:
            return "multiclass"
        else:
            return "regression"
