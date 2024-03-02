import pandas as pd
import polars as pl

from ..types import XyArrayLike


def reset_X(X: XyArrayLike, feature_names: list[str]) -> XyArrayLike:
    if isinstance(X, pd.DataFrame):
        return X[feature_names].reset_index(drop=True)
    elif isinstance(X, pl.DataFrame):
        return X.select(feature_names)
    else:
        if X.shape != len(feature_names):
            raise ValueError(f"X shape {X.shape} is not equal to feature_names length {len(feature_names)}")
        return X
