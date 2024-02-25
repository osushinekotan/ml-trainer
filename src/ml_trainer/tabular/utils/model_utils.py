import pandas as pd
import polars as pl

from src.tabular.types import XyArrayLike


def reset_X(X: XyArrayLike, feature_names: list[str]) -> XyArrayLike:
    if isinstance(X, pd.DataFrame):
        X = X[feature_names].reset_index(drop=True)
    elif isinstance(X, pl.DataFrame):
        X = X.select(feature_names)
    return X
