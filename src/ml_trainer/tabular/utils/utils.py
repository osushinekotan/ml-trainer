import hashlib
from typing import Any


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
