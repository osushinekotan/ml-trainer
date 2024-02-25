from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)
