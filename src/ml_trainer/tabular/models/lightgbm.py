import pandas as pd
from lightgbm import LGBMModel

from src.ml_trainer.tabular.models.base import EstimatorBase
from src.ml_trainer.tabular.utils.model_utils import reset_X
from src.ml_trainer.tabular.utils.utils import generate_uid


class LightGBMModel(EstimatorBase):
    """lightgbm model wrapper."""

    def __init__(
        self,
        feature_names: list[str],
        params: dict,
        fit_params: dict,
        tune: bool = False,  # TODO: implement hyperparameter tuning with optuna
        tune_params: dict = {},
    ) -> None:
        self.model = LGBMModel(**params)
        self.params = params
        self.fit_params = fit_params
        self.feature_names = feature_names

        self.tune = tune
        self.tune_params = tune_params

        self.estimator_name = "lightgbm"

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = reset_X(X_train, self.feature_names)
        X_val = reset_X(X_val, self.feature_names)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            **self.fit_params,
        )

    def predict(self, X):
        X = reset_X(X, self.feature_names)
        return self.model.predict(X)

    def get_feature_importance(self):
        assert len(self.feature_names) == len(self.model.feature_importances_)
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        )
        return importance_df

    def get_params(self):
        return self.model.get_params()

    @property
    def uid(self):
        uid = generate_uid(
            self.params,
            self.fit_params,
            self.estimator_name,
            self.tune,
            self.tune_params,
        )
        return f"{self.estimator_name}_{uid}"
