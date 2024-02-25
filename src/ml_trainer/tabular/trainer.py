# type: ignore
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, _BaseKFold

from src.ml_trainer.models.base import EstimatorBase
from src.ml_trainer.tabular.types import XyArrayLike


class Trainer:
    """Single Task Trainer for Tabular Data."""

    def __init__(
        self,
        estimators: list[EstimatorBase],  # lightgbm, etc
        ensemble: bool = False,
        # cross_validation params
        split_type: _BaseKFold | str | list[int] = KFold,  # if "fold" then use "fold" columns in the X (DataFrame)
        n_splits: int = 5,
        groups: ArrayLike | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        self.estimators = estimators
        self.ensemble = ensemble

        self.split_type = split_type
        self.n_splits = n_splits
        self.groups = groups
        self.seed = seed

        self.kwargs = kwargs
        self.fitted_estimators: dict[str:EstimatorBase] = {}
        self.is_fitted = False

    def train(
        self,
        X_train: XyArrayLike,
        y_train: XyArrayLike,
        X_val: XyArrayLike,
        y_val: XyArrayLike,
    ) -> None:
        task = self.judge_task(y_train)
        print(task)

        for estimator in self.estimators:
            estimator.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

            self.fitted_estimators[estimator.uid] = estimator

        self.is_fitted = True
        return self.fitted_estimators  # scores, feature_importance, etc

    def cv(self, X_train: XyArrayLike, y_train: XyArrayLike):
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pl.DataFrame):
            if self.split_type == "fold" and "fold" in X_train.columns:
                folds = self._generate_folds(X_train["fold"])
        elif isinstance(self.split_type, list):
            folds = self._generate_folds(self.split_type)
        else:
            folds = self.split_type(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            ).split(X_train, y_train, groups=self.groups)

        fold_fitted_estimators = {}
        for i_fold, (tr_idx, va_idx) in enumerate(folds):
            # dataframe と array に対応
            val_mask = np.zeros(len(X_train), dtype=bool)
            val_mask[va_idx] = True

            X_tr, X_va = X_train[tr_idx].copy(), X_train[va_idx].copy()
            y_tr, y_va = y_train[tr_idx].copy(), y_train[va_idx].copy()

            fitted_estimators = self.train(X_tr, y_tr, X_va, y_va)
            fold_fitted_estimators[f"fold{i_fold}"] = fitted_estimators

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
