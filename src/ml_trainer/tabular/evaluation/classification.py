from typing import Callable

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def macro_roc_auc_score(y_true: NDArray, y_pred: NDArray) -> float:
    return roc_auc_score(y_true, y_pred, average="macro")


def optimize_threshold(
    y_true: NDArray,
    y_pred: NDArray,
    metrics: Callable,
    n_jobs: int = -1,
    maximize: bool = True,
    step: float = 0.01,
) -> float:
    search_vals = list(np.arange(0, 1, step))

    def _calc(th: float) -> float:
        score = metrics(y_true, y_pred >= th)
        if not maximize:
            return score * -1
        return score

    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_calc)(th) for th in search_vals)
    max_idx = np.argmax(results)
    return search_vals[max_idx]


def opt_f1_score(
    y_true: NDArray,
    y_pred: NDArray,
) -> dict[str, float]:
    th = optimize_threshold(y_true, y_pred, metrics=f1_score)
    return {"th": th, "score": f1_score(y_true, y_pred >= th)}


def opt_acc_score(
    y_true: NDArray,
    y_pred: NDArray,
) -> dict[str, float]:
    th = optimize_threshold(y_true, y_pred, metrics=accuracy_score)
    return {"th": th, "score": accuracy_score(y_true, y_pred >= th)}
