import category_encoders as ce

from .base import FeatureTransformerBase
from .types import XyArrayLike


class OneHotEncoder(FeatureTransformerBase):
    def __init__(self, params: dict):
        self.encoder = ce.OneHotEncoder(**params)
        self.feature_transformer_name = "OneHotEncoder"

    @property
    def snapshot_items(self) -> list[str]:
        return ["encoder", "params"]

    def fit(self, X: XyArrayLike, y: XyArrayLike | None = None) -> "OneHotEncoder":
        self.encoder.fit(X, y)
        return self

    def transform(self, X: XyArrayLike) -> XyArrayLike:
        return self.encoder.transform(X)


class OrdinalEncoder(FeatureTransformerBase):
    def __init__(self, params: dict):
        self.encoder = ce.OrdinalEncoder(**params)
        self.feature_transformer_name = "OrdinalEncoder"

    @property
    def snapshot_items(self) -> list[str]:
        return ["encoder", "params"]

    def fit(self, X: XyArrayLike, y: XyArrayLike | None = None) -> "OrdinalEncoder":
        self.encoder.fit(X, y)
        return self

    def transform(self, X: XyArrayLike) -> XyArrayLike:
        return self.encoder.transform(X)
