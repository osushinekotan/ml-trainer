from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd

from src.ml_trainer.tabular.types import XyArrayLike


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

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractproperty
    def uid(self) -> str:
        pass
