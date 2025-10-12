"""Lightweight dataset utilities built on top of the math layer."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["atlas_tiny", "converters", "generators", "quantities"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin module shim
    if name in __all__:
        module = import_module(f"viterbo.datasets2.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
