"""Backwards-compatible convenience calls share a single lazy model backend."""

from functools import lru_cache


@lru_cache(maxsize=1)
def default_backend():
    from config import Config
    from models import ModelBackend

    return ModelBackend(Config())
