import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from ..tabular.utils.trainer_utils import transform_proba_to_label


def make_feature_importance_fig(
    feature_importance_df: pd.DataFrame,
    plot_type: str = "auto",
    top_n: int | None = None,
    palette: str = "bwr_r",
    title: str = "importance",
) -> Figure:
    """特徴量の重要度を可視化する.

    Args:
        feature_importance_df (pd.DataFrame): model.get_feature_importance() で得られる DataFrame
        plot_type (str, optional): グラフタイプの選択。"bar", "boxen" or "auto". Defaults to "auto".
        top_n (int | None, optional): プロットする重要度 top n. Defaults to None.
        palette (str, optional): プロット時の色. Defaults to "bwr_r".
        title (str, optional): プロット時のタイトル. Defaults to "importance".

    Returns:
        Figure:
            特徴量の重要度を可視化した Figure オブジェクト
            tree 系のモデルの場合は特徴量重要度、線形系のモデルの場合は係数をプロット
    """
    japanize_matplotlib.japanize()

    if plot_type == "auto":
        # NOTE : fold ユニーク数が 1 なら bar, 2 以上なら boxen
        plot_type = "boxen" if feature_importance_df["fold"].nunique() > 1 else "bar"

    if plot_type not in ["boxen", "bar"]:
        raise ValueError("Invalid plot_type")

    order = (
        feature_importance_df.groupby("feature").sum()[["importance"]].sort_values("importance", ascending=False).index
    )
    if top_n is not None:
        order = order[:top_n]

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.25)))
    plot_params = dict(
        data=feature_importance_df,
        x="importance",
        y="feature",
        order=order,
        ax=ax,
        hue_order=order,
        hue="feature",
        palette=palette,
        orient="h",
    )
    if plot_type == "boxen":
        sns.boxenplot(**plot_params)
    elif plot_type == "bar":
        sns.barplot(**plot_params)
    else:
        raise NotImplementedError()

    ax.tick_params(axis="x", rotation=90)
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    return fig


def make_confusion_matrix_fig(
    y_true: NDArray,
    y_pred: NDArray,
    normalize: bool = False,
    cmap: str = "bwr_r",
) -> Figure:
    """_summary_

    Args:
        y_true (ArrayLike): 正解ラベル
        y_pred (ArrayLike): 予測ラベル。予測確率 input の場合は、自動でラベルを取得。binary の場合は 0.5 で閾値を設定、それ以外は argmax でラベルを取得。
        normalize (bool, optional): 正解ラベル方向に normalize するかどうか. Defaults to False.
        cmap (str, optional): プロットの色. Defaults to "bwr_r".

    Returns:
        Figure: 混同行列を可視化した Figure オブジェクト
    """
    y_pred = transform_proba_to_label(y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # ラベルのユニークな数を取得
    num_labels = len(np.unique(y_true))

    # figsize をラベルの数に基づいて動的に調整
    # 基本のサイズを (8, 6) とし、ラベルごとに幅と高さを 1 ずつ追加
    fig_width = 8 + max(0, num_labels - 10)  # 10 以下のラベル数では基本のサイズを使用
    fig_height = 6 + max(0, num_labels - 10)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    return fig
