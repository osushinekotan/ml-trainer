# type: ignore
import json
from pathlib import Path
from typing import Callable

import japanize_matplotlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from rich.console import Console
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, roc_auc_score
from sklearn.model_selection import BaseCrossValidator, KFold

from .evaluation.classification import macro_roc_auc_score
from .evaluation.regression import root_mean_squared_error
from .models.base import EstimatorBase
from .types import XyArrayLike

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
        split_type: BaseCrossValidator
        | str
        | list[int] = KFold,  # if "fold" then use "fold" columns in the X (DataFrame)
        n_splits: int = 5,
        groups: ArrayLike | None = None,
        seed: int | None = None,
        eval_metrics: list[str] | None | str = "auto",
        custom_eval: Callable | None = None,
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
        self.eval_metrics = eval_metrics
        self.custom_eval = custom_eval

        self.kwargs = kwargs
        self.is_fitted = False
        self.is_cv = False

    def get_eval_metrics(self, task: str) -> dict[str, Callable]:
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
    ) -> dict:
        """
        Train Estimators. (複数の) Estimator を学習し、保存する。
        out_dir_est (out_dir / estimator_uid) に estimator ごとの結果が保存される。

        Args:
            X_train (XyArrayLike): 学習用の特徴量
            y_train (XyArrayLike): 学習用の目的変数
            X_val (XyArrayLike): 検証用の特徴量
            y_val (XyArrayLike): 検証用の目的変数
            out_dir (Path | None, optional): estimator や pred, scores の保存先. Defaults to None.

        Returns:
            dict: 学習結果を格納した辞書。 {estimator_uid: {"estimator": estimator, "pred": pred, "scores": scores}} の形式で出力される。
        """
        out_dir = out_dir or self.out_dir
        out_dir.mkdir(exist_ok=True, parents=True)

        task = self.judge_task(y_train)
        metrics = self.get_eval_metrics(task)

        console.print(f"Estimator saving to: {out_dir}", style="bold green")
        console.print(f"[{task} Metrics]: \n{metrics}", style="bold green")

        resutls: dict = {}
        for estimator in self.estimators:
            estimator_uid = estimator.uid
            out_dir_est = out_dir / estimator_uid
            estimator_path = out_dir_est / "estimator.pkl"

            console.print(f"[{estimator_uid}] start training :rocket:", style="bold blue")
            estimator.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            estimator.save(estimator_path)

            pred = estimator.predict(X_val)
            joblib.dump(pred, out_dir_est / "pred.pkl")

            scores = {metric_name: metric(y_true=y_val, y_pred=pred) for metric_name, metric in metrics.items()}
            json.dump(scores, open(out_dir_est / "scores.json", "w"), indent=4)

            console.print(f"[{estimator_uid}] scores: \n{json.dumps(scores, indent=4)}", style="bold blue")
            resutls[estimator_uid] = {
                "estimator": estimator,
                "pred": pred,
                "scores": scores,
            }

        if self.ensemble:
            # NOTE : mean ensemble 予測値取得する
            # TODO : weighed average や stacking など他のアンサンブル手法も実装したい
            ensemble_dir = out_dir / "ensemble"
            ensemble_dir.mkdir(exist_ok=True, parents=True)

            ensemble_pred = np.mean([_pred for _pred in resutls.values()], axis=0)
            ensemble_scores = {
                metric_name: metric(y_true=y_val, y_pred=ensemble_pred) for metric_name, metric in metrics.items()
            }
            joblib.dump(ensemble_pred, ensemble_dir / "pred.pkl")
            json.dump(ensemble_scores, open(ensemble_dir / "scores.json", "w"), indent=4)

            console.print(f"[ensemble] scores: \n{json.dumps(ensemble_scores, indent=4)}", style="bold blue")
            resutls["ensemble"] = {"pred": ensemble_pred, "scores": ensemble_scores}

        self.is_fitted = True
        return resutls

    def train_cv(self, X_train: XyArrayLike, y_train: XyArrayLike) -> dict[str, ArrayLike]:
        """Cross Validation.

        Args:
            X_train (XyArrayLike): 特徴量
            y_train (XyArrayLike): 目的変数

        Returns:
            dict[str, ArrayLike]: estimator ごとの oof 予測値
        """
        self.is_cv = True

        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pl.DataFrame):
            # NOTE : split_type が "fold" かつ X_train が DataFrame の場合は fold カラムに従って分割する splitter を作成
            if self.split_type == "fold" and "fold" in X_train.columns:
                folds = self._generate_folds(X_train["fold"])
                self.n_splits = X_train["fold"].nunique()
        elif isinstance(self.split_type, list):
            # NOTE : split_type が list の場合はそのリストに従って分割する splitter を作成
            folds = self._generate_folds(self.split_type)
            self.n_splits = len(np.unique(self.split_type))
        else:
            folds = self.split_type(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            ).split(X_train, y_train, groups=self.groups)

        task = self.judge_task(y_train)
        console.print(f"[{task}] Cross Validation: {self.n_splits} folds", style="bold green")

        metrics = self.get_eval_metrics(task)

        fold_fitted_results = {}
        for i_fold, (tr_idx, va_idx) in enumerate(folds):
            # dataframe と array に対応
            val_mask = np.zeros(len(X_train), dtype=bool)
            val_mask[va_idx] = True

            X_tr, X_va = X_train[tr_idx].copy(), X_train[va_idx].copy()
            y_tr, y_va = y_train[tr_idx].copy(), y_train[va_idx].copy()

            fitted_resuts = self.train(X_tr, y_tr, X_va, y_va, out_dir=self.out_dir / f"fold{i_fold}")
            fold_fitted_results[f"fold{i_fold}"] = fitted_resuts

        # NOTE : oof 予測値取得する eg. {est1: pred, est2: pred, ...} pred: ArrayLike
        oof = self.get_oof(fold_fitted_results)
        for est, pred in oof.items():
            result_dir = self.out_dir / "results" / est
            result_dir.mkdir(exist_ok=True, parents=True)

            # oof predictions
            joblib.dump(pred, result_dir / "pred.pkl")

            # oof scores
            scores = {metric_name: metric(y_true=y_train, y_pred=pred) for metric_name, metric in metrics.items()}
            json.dump(scores, open(result_dir / "scores.json", "w"), indent=4)
            console.print(f"[oof] [{est}] scores: \n{json.dumps(scores, indent=4)}", style="bold blue")

        return oof

    def get_oof(self, cv_results: dict) -> dict[str:ArrayLike]:
        """Get Out Of Fold Prediction results.

        Returns:
            dict: estimator ごとの oof 予測値を格納した辞書。 {est1: pred, est2: pred, ...} の形式で出力される (pred: ArrayLike)。
        """
        oof_results = {}  # {est1: pred, est2: pred, ...}

        for fold in cv_results:
            for est, data in cv_results[fold].items():
                scores = data["scores"]
                console.print(
                    f"[fold{fold}] [{est}] scores: \n{json.dumps(scores, indent=4)}",
                    style="bold blue",
                )
                if est not in oof_results:
                    oof_results[est] = list(data["pred"])
                else:
                    oof_results[est].extend(list(data["pred"]))

        return oof_results

    def predict(self, X: XyArrayLike, out_dir: Path | None = None, save: bool = False) -> dict[str, ArrayLike]:
        """Predict. (複数の) Estimator を使って予測を行う。

        Args:
            X (XyArrayLike): 予測用の特徴量
            out_dir (Path | None, optional): 学習ずみの Estimator が保存されているディレクトリ. Defaults to None.
            save (bool, optional): 予測結果を out_dir に保存するかどうか. Defaults to False.

        Returns:
            dict[str, ArrayLike]: estimator ごとの予測値を格納した辞書。 {est1: pred, est2: pred, ...} の形式で出力される (pred: ArrayLike)。
        """

        console.print(f"Estimator restoring from: {out_dir}", style="bold green")

        resutls = {}
        for estimator in self.estimators:
            estimator_uid = estimator.uid
            estimator_dir = out_dir / estimator_uid

            console.print(f"[{estimator_uid}] start predicting :rocket:", style="bold blue")
            estimator.load(estimator_dir / "estimator.pkl")

            pred = estimator.predict(X)
            resutls[estimator_uid] = pred
            if save:
                joblib.dump(pred, estimator_dir / "test_pred.pkl")

        if self.ensemble:
            # NOTE : mean ensemble 予測値取得する
            ensemble_pred = np.mean([_pred for _pred in resutls.values()], axis=0)
            resutls["ensemble"] = ensemble_pred
            if save:
                joblib.dump(pred, out_dir / "ensemble" / "test_pred.pkl")

        return resutls

    def predict_cv(self, X: XyArrayLike, out_dir: Path | None = None, save: bool = False) -> dict[str, ArrayLike]:
        """Predict. (複数の) Estimator を使って予測を行う。

        Args:
            X (XyArrayLike): 予測用の特徴量
            out_dir (Path | None, optional): 学習ずみの Estimator が保存されているディレクトリ. Defaults to None.
            save (bool, optional): 予測結果を out_dir に保存するかどうか. Defaults to False.

        Returns:
            dict[str, ArrayLike]: estimator ごとの予測値を格納した辞書。fold ごとの平均予測値が出力される。
        """
        console.print(f"Predict Cross Validation : {self.n_splits} folds.", style="bold green")

        fold_results = {}
        for i_fold in range(self.n_splits):
            fold_dir = out_dir / f"fold{i_fold}"

            pred_results = self.predict(X, out_dir=fold_dir, save=False)  # NOTE : cv 予測時は各 fold pred は保存しない
            fold_results[f"fold{i_fold}"] = pred_results

        fold_means = self.get_fold_mean(fold_results)
        if save:
            for est, pred in fold_means.items():
                result_dir = out_dir / "results" / est
                result_dir.mkdir(exist_ok=True, parents=True)
                joblib.dump(pred, result_dir / "test_pred.pkl")
        return fold_means

    def get_fold_mean(self, fold_results: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
        """Get Aggregated Prediction results.

        Returns:
            dict: estimator ごとの fold ごとの予測値を集約した辞書。 {est1: pred, est2: pred, ...} の形式で出力される (pred: ArrayLike)。
        """
        # 各 est の fold ごとの pred 値を収集
        fold_values = {est: [] for est in fold_results[fold_results.keys()][0]}
        for fold in fold_results:
            for est, data in fold_results[fold].items():
                fold_values[est].append(data["pred"])

        # 各 est の fold 平均を計算
        fold_means = {}
        for est, values in fold_values.items():
            mean_values = np.mean(values, axis=0).tolist()  # mean 予測値
            fold_means[est] = mean_values

        return fold_means

    def make_plot_feature_importances(
        self,
        out_dir: Path | None = None,
        top_n: int | None = 50,
        save: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Get Feature Importance.

        Args:
            out_dir (Path | None, optional): 学習ずみの Estimator が保存されているディレクトリ. Defaults to None.
            top_n (int | None, optional): 上位何個の feature importance をプロットするか。None なら全てプロットする。 Defaults to None.
            save (bool, optional): 予測結果を out_dir に保存するかどうか. Defaults to True.

        Returns:
            dict[str, pd.DataFrame]:
                estimator ごとの feature importance を格納した辞書。 {est1: importance_df, est2: importance_df, ...} の形式で出力される。
                importance_df は feature_name, importance, fold のカラムを持つ DataFrame。
        """
        if not self.is_fitted:
            raise ValueError("Estimator is not fitted yet.")

        out_dir = out_dir or self.out_dir
        n_splits = self.n_splits if self.is_cv else 1

        feature_importances = {est.uid: pd.DataFrame() for est in self.estimators}
        for i_fold in range(n_splits):
            if not self.is_cv:
                result_dir = out_dir
            else:
                result_dir = out_dir / f"fold{i_fold}"

            for estimator in self.estimators:
                estimator_uid = estimator.uid
                estimator_dir = result_dir / estimator_uid
                estimator.load(estimator_dir / "estimator.pkl")

                importance_df = estimator.get_feature_importance()
                importance_df["fold"] = i_fold  # not is_cv なら fold は 0 にする

                feature_importances[estimator_uid] = pd.concat(
                    [feature_importances[estimator_uid], importance_df], axis=0
                )

        # NOTE : plot feature importance
        for est, df in feature_importances.items():
            fig = self.make_feature_importance_fig(df, plot_type="auto", top_n=top_n)
            fig.show()
            if save:
                save_dir = out_dir / "results" / est
                df.to_csv(save_dir / "feature_importance.csv", index=False)
                fig.savefig(save_dir / "feature_importance.png", dpi=300)

        return feature_importances

    def make_feature_importance_fig(
        self,
        feature_importance_df: pd.DataFrame,
        plot_type: str = "auto",
        top_n: int | None = None,
    ) -> Figure | pd.DataFrame:
        japanize_matplotlib.japanize()

        if plot_type == "auto":
            # NOTE : fold ユニーク数が 1 なら bar, 2 以上なら boxen
            plot_type = "boxen" if feature_importance_df["fold"].nunique() > 1 else "bar"

        if plot_type not in ["boxen", "bar"]:
            raise ValueError("Invalid plot_type")

        order = (
            feature_importance_df.groupby("feature")
            .sum()[["importance"]]
            .sort_values("importance", ascending=False)
            .index
        )
        if top_n is not None:
            order = order[:top_n] or order

        fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.25)))
        plot_params = dict(
            data=feature_importance_df,
            x="importance",
            y="feature",
            order=order,
            ax=ax,
            palette="viridis",
            orient="h",
        )
        if plot_type == "boxen":
            sns.boxenplot(**plot_params)
        elif plot_type == "bar":
            sns.barplot(**plot_params)
        else:
            raise NotImplementedError()

        ax.tick_params(axis="x", rotation=90)
        ax.set_title("Importance")
        ax.grid()
        fig.tight_layout()
        return fig

    @property
    def snapshot_items(self):
        return [
            "estimators",
            "ensemble",
            "out_dir",
            "feature_names",
            "estimator_name",
            "split_type",
            "n_splits",
            "groups",
            "seed",
            "kwargs",
            "is_fitted",
            "is_cv",
            "eval_metrics",
            "custom_eval",
        ]

    def save(self, filepath: Path) -> None:
        """snapshot items を保存する."""
        filepath.parent.mkdir(exist_ok=True, parents=True)
        snapshot = tuple([getattr(self, item) for item in self.snapshot_items])
        joblib.dump(snapshot, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "Trainer":
        """保存した snapshot items を classmethod で読み込む.
        >>> trainer = Trainer.load("trainer.pkl")
        >>> trainer.predict(X)
        """
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        snapshot = joblib.load(filepath)
        for item, value in zip(cls.snapshot_items, snapshot):
            setattr(cls, item, value)
        return cls

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
