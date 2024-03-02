import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
)

from ..types import XyArrayLike
from ..utils.model_utils import reset_X
from .base import EstimatorBase


class LinearRegressionModel(EstimatorBase):
    """sklearn linear regression model wrapper."""

    def __init__(
        self,
        feature_names: list[str],
        params: dict = {},
        fit_params: dict = {},
        estimator_name: str = "ridge",
    ) -> None:
        self.estimator_name = estimator_name

        if self.estimator_name == "linear_regression":
            self.model = LinearRegression(**params)
        elif self.estimator_name == "ridge":
            self.model = Ridge(**params)
        elif self.estimator_name == "lasso":
            self.model = Lasso(**params)
        elif self.estimator_name == "bayesian_ridge":
            self.model = BayesianRidge(**params)
        elif self.estimator_name == "elastic_net":
            self.model = ElasticNet(**params)
        elif self.estimator_name == "huber":
            self.model = HuberRegressor(**params)
        elif self.estimator_name == "quantile":
            self.model = QuantileRegressor(**params)
        elif self.estimator_name == "logistic_regression":
            self.model = LogisticRegression(**params)
        else:
            raise ValueError("Invalid estimator name")

        self.params = params
        self.fit_params = fit_params
        self.feature_names = feature_names

        self.uid = self.make_uid()

    def fit(self, X_train: XyArrayLike, y_train: XyArrayLike, X_val: XyArrayLike, y_val: XyArrayLike) -> None:
        X_train = reset_X(X_train, self.feature_names)
        self.model.fit(
            X_train,
            y_train,
            **self.fit_params,
        )

    def predict(self, X: XyArrayLike) -> ArrayLike:
        """ロジスティック回帰の場合は確率を返す (binary  の場合は 1 の確率)。それ以外は予測値を返す。"""

        X = reset_X(X, self.feature_names)
        if self.estimator_name == "logistic_regression":
            preds = self.model.predict_proba(X)
            if preds.shape[1] == 2:
                return preds[:, 1]
            return preds

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Feature importance はないので係数を返す。"""
        assert len(self.feature_names) == len(self.model.coef_)
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.coef_,
            }
        )
        return importance_df
