import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_preprocessing

from .base import FeatureTransformerBase
from .types import XyArrayLike


class StandardScaler(FeatureTransformerBase):
    def __init__(self, params: dict = {}):
        self.params = params.copy()
        self.exclude_columns = params.pop("exclude_columns", [])
        self.scaler = sk_preprocessing.StandardScaler(**params)
        self.feature_transformer_name = "StandardScaler"

    @property
    def snapshot_items(self) -> list[str]:
        return ["scaler", "params"]

    def fit(self, X: XyArrayLike, y: XyArrayLike | None = None) -> "StandardScaler":
        use_columns = self._get_use_columns(X)
        self.scaler.fit(X[use_columns], y)  # type: ignore
        return self

    def transform(self, X: XyArrayLike) -> XyArrayLike:
        use_columns = self._get_use_columns(X)
        if isinstance(X, pd.DataFrame):
            return self._transform_pandas(X, use_columns)
        elif isinstance(X, np.ndarray):
            return self._transform_array(X, use_columns)  # type: ignore
        else:
            raise ValueError(f"Unexpected type: {type(X)}")

    def _get_use_columns(self, X: XyArrayLike) -> list[str | int]:
        if isinstance(X, pd.DataFrame):
            raw_columns = sorted(list(X.columns))
            use_columns = sorted(list(set(raw_columns) - set(self.exclude_columns)))
            return use_columns

        elif isinstance(X, np.ndarray):
            return [i for i in range(X.shape[1]) if i not in self.exclude_columns]

        else:
            raise ValueError(f"Unexpected type: {type(X)}")

    def _transform_pandas(self, X: pd.DataFrame, use_columns: list[str | int]) -> pd.DataFrame:
        X_ = X[use_columns]
        X_ = pd.DataFrame(self.scaler.transform(X_), columns=use_columns)
        return pd.concat([X_, X[self.exclude_columns]], axis=1)  # scale していない列を結合

    def _transform_array(self, X: np.ndarray, use_columns: list[int]) -> np.ndarray:
        X_ = X[:, use_columns]
        X_ = self.scaler.transform(X_)
        # 列の順番を保持して結合
        for i in self.exclude_columns:
            X_ = np.insert(X_, i, X[:, i], axis=1)
        return X_  # TODO: 列の順番を保持するか確認
