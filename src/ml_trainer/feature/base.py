import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib

from .types import XyArrayLike


# TODO : trainer と共通のものにする
def generate_uid(*args: Any) -> str:
    """Generate a unique identifier from args, handling objects consistently."""

    def obj_repr(obj: Any) -> str:
        """Return a consistent string representation for objects."""
        if callable(obj):
            return f"{obj.__module__}.{obj.__qualname__}"
        elif hasattr(obj, "__dict__"):
            return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}:{obj.__dict__}"
        else:
            return str(obj)

    unique_string = "_".join(map(obj_repr, args))
    return hashlib.md5(unique_string.encode()).hexdigest()


class FeatureTransformerBase(ABC):
    @abstractmethod
    def fit(self, X: XyArrayLike, y: XyArrayLike | None = None) -> "FeatureTransformerBase":
        pass

    @abstractmethod
    def transform(self, X: XyArrayLike) -> XyArrayLike:
        pass

    @property
    def snapshot_items(self) -> list:
        return ["params"]

    def make_uid(self) -> str:
        uid_sources = getattr(self, "params")  # params のみで uid を生成する
        base_uid = generate_uid(*uid_sources)
        feature_transformer_name = getattr(self, "feature_transformer_name")
        return f"{feature_transformer_name}_{base_uid}"

    def save(self, filepath: Path) -> None:
        """snapshot items を保存する."""
        filepath.parent.mkdir(exist_ok=True, parents=True)
        snapshot = tuple([getattr(self, item) for item in self.snapshot_items])
        joblib.dump(snapshot, filepath)

    def load(self, filepath: Path) -> None:
        """保存したsnapshot itemsを読み込む.
        e.g. snapshot items: (model, feature_names, params, fit_params)
        >>> feature_encoder.load("model.pkl")
        >>> feature_encoder.predict(X)
        """
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        snapshot = joblib.load(filepath)
        for item, value in zip(self.snapshot_items, snapshot):
            setattr(self, item, value)
