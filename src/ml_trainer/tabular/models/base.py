from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd
from numpy.typing import ArrayLike

from ..types import XyArrayLike
from ..utils.utils import generate_uid


class EstimatorBase(ABC):
    """abstract class for all models."""

    # NOTE : 継承先の initializer で設定する. e.g. self.uid = self.make_uid()
    # すべての派生クラスで uid が使用される
    uid: str  # unique identifier for the model
    use_cache: bool  # whether to use cache or not

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
    def predict(self, X: XyArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass

    def make_uid(self) -> str:
        uid_sources = [getattr(self, item) for item in self.snapshot_items if item != "model"]
        base_uid = generate_uid(*uid_sources)
        estimator_name = getattr(self, "estimator_name")
        return f"{estimator_name}_{base_uid}"

    @property
    def snapshot_items(self) -> list:
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
