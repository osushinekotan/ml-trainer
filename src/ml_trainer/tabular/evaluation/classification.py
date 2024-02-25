from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score


def macro_roc_auc_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return roc_auc_score(y_true, y_pred, average="macro")
