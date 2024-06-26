import json
from logging import Logger
from pathlib import Path
from typing import Callable, Generator

import joblib
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils.multiclass import type_of_target

from .evaluation.classification import (
    confusion_matrix_list,
    macro_roc_auc_score,
    macro_roc_ovr_score,
    opt_acc_score,
    opt_balanced_acc_score,
    opt_f1_score,
)
from .models.base import EstimatorBase
from .types import XyArrayLike
from .visualization import (
    make_calibration_curve_fig,
    make_confusion_matrix_fig,
    make_distribution_fig,
    make_feature_importance_fig,
    make_pecision_recall_curve_fig,
)

REGRESSION_METRICS = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
}

BINARY_METRICS = {
    "loglogss": log_loss,
    "auc": macro_roc_auc_score,
    "opt_acc": opt_acc_score,
    "opt_balanced_acc": opt_balanced_acc_score,
    "opt_f1": opt_f1_score,
    "average_precision": lambda y_true, y_pred: average_precision_score(y_true, y_pred),
}

MULTICLASS_METRICS = {
    "macro_auc": macro_roc_ovr_score,
    "loglogss": log_loss,
    "confusion_matrix": confusion_matrix_list,
}

TASK_TYPES = ["binary", "multiclass", "regression"]


def log(message: str, logger: Logger | None = None) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message)


class Trainer:
    """Single Task Trainer for Tabular Data."""

    snapshot_items: list[str] = [
        "estimators",
        "ensemble",
        "out_dir",
        "split_type",
        "n_splits",
        "groups",
        "seed",
        "is_fitted",
        "is_cv",
        "eval_metrics",
        "custom_eval",
        "task_type",
        "trainer_name",
        "scores_df",
    ]

    def __init__(
        self,
        estimators: list[EstimatorBase] | None = None,  # lightgbm, etc
        ensemble: bool = False,
        out_dir: str | Path | None = None,
        # cross_validation params
        split_type: BaseCrossValidator
        | str
        | list[int] = KFold,  # if "fold" then use "fold" columns in the X (DataFrame)
        n_splits: int = 5,
        groups: ArrayLike | None = None,
        seed: int | None = None,
        trainer_name: str = "trainer",
        task_type: str = "auto",
        eval_metrics: list[str] | None | str = "auto",
        custom_eval: Callable | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.estimators = estimators
        self.ensemble = ensemble

        if out_dir is None:
            self.out_dir = Path.cwd() / "output"
        else:
            self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.split_type = split_type
        self.n_splits = n_splits
        self.groups = groups
        self.seed = seed
        self.eval_metrics = eval_metrics
        self.custom_eval = custom_eval

        self.is_fitted = False
        self.is_cv = False
        self.scores_df = pd.DataFrame()

        self.task_type = task_type
        self.trainer_name = trainer_name
        self.logger = logger

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
        if self.estimators is None:
            raise ValueError("estimators must be specified")

        out_dir = out_dir or self.out_dir
        out_dir.mkdir(exist_ok=True, parents=True)

        task = self._judge_task(y_train)
        metrics = self.get_eval_metrics(task)

        log(f"Estimator saving to: {out_dir}", logger=self.logger)
        log(rf"{task} metrics: {[metric.__name__ for metric in metrics.values()]}", logger=self.logger)

        results: dict = {}
        for estimator in self.estimators:
            estimator_uid = estimator.uid
            out_dir_est = out_dir / estimator_uid
            estimator_path = out_dir_est / "estimator.pkl"

            if estimator.use_cache and estimator_path.exists():
                # NOTE : use_cache かつ既に学習済みの場合はロードする
                log(f"Estimator restoring from: {estimator_path}", logger=self.logger)
                estimator.load(estimator_path)
                pred = joblib.load(out_dir_est / "pred.pkl")
                scores = json.load(open(out_dir_est / "scores.json", "r"))
                results[estimator_uid] = {
                    "estimator": estimator,
                    "pred": pred,
                    "scores": scores,
                }
                continue

            log(rf"[{estimator_uid}] start training 🚀", logger=self.logger)
            estimator.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
            estimator.save(estimator_path)

            pred = estimator.predict(X_val)
            joblib.dump(pred, out_dir_est / "pred.pkl")

            scores = {metric_name: metric(y_true=y_val, y_pred=pred) for metric_name, metric in metrics.items()}
            json.dump(scores, open(out_dir_est / "scores.json", "w"), indent=4)

            log(rf"[{estimator_uid}] scores: {json.dumps(scores, indent=4)}", logger=self.logger)
            results[estimator_uid] = {
                "estimator": estimator,
                "pred": pred,
                "scores": scores,
            }

        if self.ensemble:
            # NOTE : mean ensemble 予測値取得する
            # TODO : weighed average や stacking など他のアンサンブル手法も実装したい
            ensemble_dir = out_dir / "ensemble"
            ensemble_dir.mkdir(exist_ok=True, parents=True)

            ensemble_pred = np.mean([data["pred"] for data in results.values()], axis=0)
            ensemble_scores = {
                metric_name: metric(y_true=y_val, y_pred=ensemble_pred) for metric_name, metric in metrics.items()
            }
            joblib.dump(ensemble_pred, ensemble_dir / "pred.pkl")
            json.dump(ensemble_scores, open(ensemble_dir / "scores.json", "w"), indent=4)

            log(f"[ensemble] scores: {json.dumps(ensemble_scores, indent=4)}", logger=self.logger)
            results["ensemble"] = {"pred": ensemble_pred, "scores": ensemble_scores}

        self.is_fitted = True
        self.scores_df = pd.concat(
            [self.scores_df, self.make_socres_df(results, name=out_dir.name)],
        ).reset_index(drop=True)  # cross validation に対応するため concat する
        return results

    def train_cv(self, X_train: XyArrayLike, y_train: XyArrayLike) -> dict[str, ArrayLike]:
        """Cross Validation.

        Args:
            X_train (XyArrayLike): 特徴量
            y_train (XyArrayLike): 目的変数

        Returns:
            dict[str, ArrayLike]: estimator ごとの oof 予測値
        """
        self.is_cv = True

        # NOTE : split_type が "fold" かつ X_train が DataFrame の場合は fold カラムに従って分割する splitter を作成
        if self.split_type == "fold":
            assert isinstance(X_train, pd.DataFrame) or isinstance(X_train, pl.DataFrame), "X_train must be DataFrame"
            assert "fold" in X_train.columns, "X_train must have 'fold' column"
            folds = self._generate_folds(X_train["fold"])
            self.n_splits = X_train["fold"].nunique()  # type: ignore
        elif isinstance(self.split_type, list):
            # NOTE : split_type が list の場合はそのリストに従って分割する splitter を作成
            folds = self._generate_folds(self.split_type)
            self.n_splits = len(np.unique(self.split_type))
        elif issubclass(self.split_type, BaseCrossValidator):  # type: ignore
            folds = self.split_type(  # type: ignore
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            ).split(X_train, y_train, groups=self.groups)
        else:
            raise ValueError(f"Invalid split_type: {self.split_type}")

        task = self._judge_task(y_train)
        log(rf"[{task}] Cross Validation: {self.n_splits} folds", logger=self.logger)

        metrics = self.get_eval_metrics(task)

        fold_fitted_results = {}
        va_orders = []
        fold_orders = []
        for i_fold, (tr_idx, va_idx) in enumerate(folds):
            # dataframe と array に対応
            val_mask = np.zeros(len(X_train), dtype=bool)
            val_mask[va_idx] = True

            X_tr, y_tr = X_train[~val_mask].copy(), y_train[~val_mask].copy()  # type: ignore
            X_va, y_va = X_train[val_mask].copy(), y_train[val_mask].copy()  # type: ignore

            fitted_resuts = self.train(X_tr, y_tr, X_va, y_va, out_dir=self.out_dir / f"fold{i_fold}")
            fold_fitted_results[f"fold{i_fold}"] = fitted_resuts
            va_orders.extend(list(va_idx))
            fold_orders.extend([i_fold] * len(va_idx))

        # NOTE : oof 予測値取得する eg. {est1: pred, est2: pred, ...} pred: ArrayLike
        oof = self.get_oof(fold_fitted_results, orders=va_orders)
        oof_results = {}
        for est, pred in oof.items():
            result_dir = self.out_dir / "results" / est
            result_dir.mkdir(exist_ok=True, parents=True)

            # oof predictions
            joblib.dump(pred, result_dir / "pred.pkl")

            # oof scores
            scores = {metric_name: metric(y_true=y_train, y_pred=pred) for metric_name, metric in metrics.items()}
            json.dump(scores, open(result_dir / "scores.json", "w"), indent=4)
            log(rf"[oof] [{est}] scores: {json.dumps(scores, indent=4)}", logger=self.logger)
            oof_results[est] = {"scores": scores}  # NOTE : make_socres_df で使うため dict で格納

        self.scores_df = pd.concat(
            [self.scores_df, self.make_socres_df(oof_results, name="oof")],
        ).reset_index(drop=True)

        if task in ["regression", "binary"]:
            # NOTE : 予測と target が一次元の場合は分布をプロットする (regression, binary)
            # TODO : train_cv 外に出す
            self.make_plot_distribution(
                oof=oof,
                y=y_train,
                fold=np.array(fold_orders)[np.argsort(va_orders)],  # type: ignore,
            )

        return oof

    def predict(self, X: XyArrayLike, out_dir: Path | None = None, save: bool = False) -> dict[str, ArrayLike]:
        """Predict. (複数の) Estimator を使って予測を行う。

        Args:
            X (XyArrayLike): 予測用の特徴量
            out_dir (Path | None, optional): 学習ずみの Estimator が保存されているディレクトリ. Defaults to None.
            save (bool, optional): 予測結果を out_dir に保存するかどうか. Defaults to False.

        Returns:
            dict[str, ArrayLike]: estimator ごとの予測値を格納した辞書。 {est1: pred, est2: pred, ...} の形式で出力される (pred: ArrayLike)。
        """
        if self.estimators is None:
            raise ValueError("estimators must be specified")

        log(f"Estimator restoring from: {out_dir}", logger=self.logger)
        out_dir = out_dir or self.out_dir

        resutls = {}
        for estimator in self.estimators:
            estimator_uid = estimator.uid
            estimator_dir = out_dir / estimator_uid

            log(rf"[{estimator_uid}] start predicting 🚀", logger=self.logger)
            estimator.load(estimator_dir / "estimator.pkl")

            pred: ArrayLike = estimator.predict(X)
            resutls[estimator_uid] = pred
            if save:
                joblib.dump(pred, estimator_dir / "test_pred.pkl")

        if self.ensemble:
            # NOTE : mean ensemble 予測値取得する
            ensemble_pred = np.mean([_pred for _pred in resutls.values()], axis=0)  # type: ignore
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
        log(f"Predict Cross Validation : {self.n_splits} folds.", logger=self.logger)
        out_dir = out_dir or self.out_dir

        fold_results = {}
        for i_fold in range(self.n_splits):
            fold_dir = out_dir / f"fold{i_fold}"

            pred_results = self.predict(X, out_dir=fold_dir, save=False)  # NOTE : cv 予測時は各 fold pred は保存しない
            fold_results[f"fold{i_fold}"] = pred_results

        fold_means = self.get_fold_mean(fold_results)  # type: ignore
        if save:
            for est, pred in fold_means.items():
                result_dir = out_dir / "results" / est
                result_dir.mkdir(exist_ok=True, parents=True)
                joblib.dump(pred, result_dir / "test_pred.pkl")
        return fold_means

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

    def make_socres_df(self, results: dict, name: str) -> pd.DataFrame:
        """Make scores DataFrame.

        Args:
            results (dict): 学習結果を格納した辞書。 {estimator_uid: {"estimator": estimator, "pred": pred, "scores": scores}} の形式。
            out_dir (Path): 保存先のディレクトリ。`name` をカラムに追加する。

        Returns:
            pd.DataFrame: scores を格納した DataFrame。
        """
        scores_df = pd.DataFrame()
        for est, data in results.items():
            scores_df = pd.concat([scores_df, pd.Series(data["scores"], name=est).to_frame().T])

        scores_df = scores_df.assign(name=name).reset_index().rename(columns={"index": "estimator"})
        return scores_df

    def get_oof(self, cv_results: dict, orders: ArrayLike | None) -> dict[str, ArrayLike]:
        """Get Out Of Fold Prediction results.

        Returns:
            dict: estimator ごとの oof 予測値を格納した辞書。 {est1: pred, est2: pred, ...} の形式で出力される (pred: ArrayLike)。
        """
        oof_results = {}  # {est1: pred, est2: pred, ...}

        for fold in cv_results:
            for est, data in cv_results[fold].items():
                scores = data["scores"]
                log(rf"[{fold}] [{est}] scores: {json.dumps(scores, indent=4)}", logger=self.logger)
                if est not in oof_results:
                    oof_results[est] = list(data["pred"])
                else:
                    oof_results[est].extend(list(data["pred"]))

        if orders is not None:
            for est, pred in oof_results.items():
                # order に従って並び替え
                oof_results[est] = np.array(pred)[np.argsort(orders)]  # type: ignore
            return oof_results  # type: ignore

        return oof_results  # type: ignore

    def get_fold_mean(self, fold_results: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
        """Get Aggregated Prediction results.

        Returns:
            dict: estimator ごとの fold ごとの予測値を集約した辞書。 {est1: pred, est2: pred, ...} の形式で出力される (pred: ArrayLike)。
        """
        # 各 est の fold ごとの pred 値を収集
        fold_values = {est: [] for est in fold_results[list(fold_results.keys())[0]]}  # type: ignore
        for fold in fold_results:
            for est, data in fold_results[fold].items():  # type: ignore
                fold_values[est].append(data)

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
        palette: str = "bwr_r",
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
        if self.estimators is None:
            raise ValueError("estimators must be specified")

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
            fig = make_feature_importance_fig(
                df,
                plot_type="auto",
                top_n=top_n,
                palette=palette,
                title=est,
            )
            if save:
                save_dir = out_dir / "results" / est
                df.to_csv(save_dir / "feature_importance.csv", index=False)
                fig.savefig(save_dir / "feature_importance.png", dpi=300)

        return feature_importances

    def make_plot_confusion_matrix(
        self,
        y: NDArray,
        out_dir: Path | None = None,
        palette: str = "GnBu",
        save: bool = True,
        threshold: float = 0.5,
        normalize: bool = True,
    ) -> None:
        if self.estimators is None:
            raise ValueError("estimators must be specified")

        if not self.is_fitted:
            raise ValueError("Estimator is not fitted yet.")

        out_dir = out_dir or self.out_dir
        if self.is_cv:
            # NOTE : cv 予測時は oof の confusion matrix をプロットする
            out_dir = out_dir / "results"
        log(f"Load probability from: {out_dir}", logger=self.logger)

        for estimator in self.estimators:
            estimator_uid = estimator.uid

            estimator_dir = out_dir / estimator_uid
            pred = joblib.load(estimator_dir / "pred.pkl")
            fig = make_confusion_matrix_fig(
                y_true=y,
                y_pred=pred,
                normalize=normalize,
                cmap=palette,
                title=estimator_uid,
                threshold=threshold,
            )
            if save:
                prefix = "normalized_" if normalize else ""
                fig.savefig(estimator_dir / f"{prefix}confusion_matrix.png", dpi=300)

    def make_plot_precision_recall_curve(
        self,
        y: NDArray,
        out_dir: Path | None = None,
        save: bool = True,
    ) -> None:
        if self.estimators is None:
            raise ValueError("estimators must be specified")

        if not self.is_fitted:
            raise ValueError("Estimator is not fitted yet.")

        out_dir = out_dir or self.out_dir
        if self.is_cv:
            # NOTE : cv 予測時は oof の precision recall curve をプロットする
            out_dir = out_dir / "results"
        log(f"Load probability from: {out_dir}", logger=self.logger)

        for estimator in self.estimators:
            estimator_uid = estimator.uid

            estimator_dir = out_dir / estimator_uid
            pred = joblib.load(estimator_dir / "pred.pkl")
            fig = make_pecision_recall_curve_fig(
                y_true=y,
                y_pred=pred,
                title=estimator_uid,
            )
            if save:
                fig.savefig(estimator_dir / "precision_recall_curve.png", dpi=300)

    def make_plot_calibration_curve(
        self,
        y: NDArray,
        n_bins: int = 10,
        out_dir: Path | None = None,
        save: bool = True,
    ) -> None:
        if self.estimators is None:
            raise ValueError("estimators must be specified")

        if not self.is_fitted:
            raise ValueError("Estimator is not fitted yet.")

        out_dir = out_dir or self.out_dir
        if self.is_cv:
            # NOTE : cv 予測時は oof の calibration curve をプロットする
            out_dir = out_dir / "results"
        log(f"Load probability from: {out_dir}", logger=self.logger)

        for estimator in self.estimators:
            estimator_uid = estimator.uid

            estimator_dir = out_dir / estimator_uid
            pred = joblib.load(estimator_dir / "pred.pkl")
            fig = make_calibration_curve_fig(y_true=y, y_pred=pred, title=estimator_uid, n_bins=n_bins)
            if save:
                fig.savefig(estimator_dir / "calibration_curve.png", dpi=300)

    def make_plot_distribution(
        self,
        oof: dict[str, ArrayLike],
        y: ArrayLike,
        fold: ArrayLike,
        result_dir: Path | None = None,
        save: bool = True,
    ) -> None:
        if result_dir is None:
            result_dir = self.out_dir / "results"

        for est, pred in oof.items():
            df = pd.DataFrame({"fold": fold, "pred": pred, "y": y})

            estimator_dir = result_dir / est
            fig = make_distribution_fig(
                df=df,
                y_col="y",
                pred_col="pred",
                fold_col="fold",
                title=est,
            )
            if save:
                fig.savefig(estimator_dir / "distribution.png", dpi=300)

    def save(self, out_dir: Path | None = None) -> None:
        """snapshot items を保存する."""
        out_dir = out_dir or self.out_dir
        out_dir.mkdir(exist_ok=True, parents=True)

        snapshot = tuple([getattr(self, item) for item in self.snapshot_items])
        joblib.dump(snapshot, out_dir / f"{self.trainer_name}.pkl")

        # scores_df もここで保存
        self.scores_df.to_csv(out_dir / "scores.csv", index=False)

    @classmethod
    def load(cls, filepath: Path) -> "Trainer":
        """保存した snapshot items を classmethod で読み込む.
        >>> trainer = Trainer.load("trainer.pkl")
        >>> trainer.predict(X)
        """
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        snapshot: list = joblib.load(filepath)
        instance = cls()  # default 引数で初期化

        for item, value in zip(cls.snapshot_items, snapshot):
            setattr(instance, item, value)
        return instance  # インスタンス化したものを返す

    def _generate_folds(self, fold_list: list | pd.Series) -> Generator:
        fold_series = pd.Series(fold_list)
        for fold in fold_series.unique():
            test_idx = fold_series[fold_series == fold].index
            train_idx = fold_series[fold_series != fold].index
            yield train_idx, test_idx

    def _judge_task(self, y: ArrayLike) -> str:
        if self.task_type != "auto":
            if self.task_type not in TASK_TYPES:
                raise ValueError(f"Invalid task type: {self.task_type}")
            return self.task_type

        # NOTE : auto なら自動で task type を判定
        task_type = type_of_target(y)
        if task_type == "binary":
            return "binary"
        elif task_type == "multiclass":
            return "multiclass"
        elif task_type == "continuous":
            return "regression"
        else:
            raise ValueError(f"Invalid task type: {task_type}")
