from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml_trainer.tabular.types import XyArrayLike
from src.ml_trainer.tabular.utils.utils import generate_uid


class EstimatorBase(ABC):
    """abstract class for all models."""

    @abstractmethod
    def fit(
        self,
        X_train: XyArrayLike,
        y_train: XyArrayLike,
        X_val: XyArrayLike,
        y_val: XyArrayLike,
    ) -> None:
        pass

    @abstractmethod
    def predict(self, X: XyArrayLike) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass

    @property
    def uid(self) -> str:
        uid_sources = [getattr(self, item) for item in self.snapshot_items]
        base_uid = generate_uid(*uid_sources)
        estimator_name = getattr(self, "estimator_name")
        return f"{estimator_name}_{base_uid}"

    @property
    def snapshot_items(self):
        return [
            "model",
            "params",
            "fit_params",
            "feature_names",
            "estimator_name",
        ]

    def save(self, filepath: Path) -> None:
        """snapshot items を保存する."""
        filepath.parent.mkdir(exist_ok=True, parents=True)
        snapshot = tuple([getattr(self, item) for item in self.snapshot_items])
        joblib.dump(snapshot, filepath)

    def load(self, filepath: Path) -> None:
        """保存したsnapshot itemsを読み込む.
        e.g. snapshot items: (model, feature_names, params, fit_params)
        >>> estimator.load("model.pkl")
        >>> estimator.predict(X)
        """
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        snapshot = joblib.load(filepath)
        for item, value in zip(self.snapshot_items, snapshot):
            setattr(self, item, value)
