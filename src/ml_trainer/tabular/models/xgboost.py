import pandas as pd
from numpy.typing import ArrayLike
from xgboost import XGBModel

from ..types import XyArrayLike
from ..utils.model_utils import reset_X
from .base import EstimatorBase


class XGBoostModel(EstimatorBase):
    """
    xgboost model wrapper.
    - xgboost parameter docs : https://xgboost.readthedocs.io/en/stable/parameter.html
    """

    def __init__(
        self,
        feature_names: list[str],
        params: dict = {},
        fit_params: dict = {},
        estimator_name: str = "xgboost",
        use_cache: bool = False,
    ) -> None:
        self.model = XGBModel(**params)
        self.params = params
        self.fit_params = fit_params
        self.feature_names = feature_names
        self.estimator_name = estimator_name
        self.use_cache = use_cache

        self.uid = self.make_uid()

    def fit(self, X_train: XyArrayLike, y_train: XyArrayLike, X_val: XyArrayLike, y_val: XyArrayLike) -> None:
        X_train = reset_X(X_train, self.feature_names)
        X_val = reset_X(X_val, self.feature_names)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            **self.fit_params,
        )

    def predict(self, X: XyArrayLike) -> ArrayLike:
        X = reset_X(X, self.feature_names)
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        assert len(self.feature_names) == len(self.model.feature_importances_)
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        )
        return importance_df
