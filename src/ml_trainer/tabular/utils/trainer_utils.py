import numpy as np
from numpy.typing import NDArray
from sklearn.utils.multiclass import type_of_target


def transform_proba_to_label(proba: NDArray, threshold: float = 0.5) -> NDArray:
    """
    予測確率を label に変換する。
    binary の場合は 0.5 で閾値を設定、それ以外は argmax でラベルを取得。label だと思われる場合はそのまま返す。
    """
    if proba.ndim > 1:
        return np.argmax(proba, axis=1)

    if type_of_target(proba) == "binary":
        return (proba >= threshold).astype(int)

    return proba
