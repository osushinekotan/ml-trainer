import hashlib


def generate_uid(*args) -> str:  # type: ignore
    """Generate unique identifier from args."""
    return hashlib.md5("_".join(map(str, args)).encode()).hexdigest()
