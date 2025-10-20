"""C++ extension scaffold via torch.utils.cpp_extension.

Provides lazy loaders for example ops implemented in C++.
If build/import fails, pure-PyTorch fallbacks are used.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_this_dir = Path(__file__).parent
_SAFE_EXCEPTIONS = (OSError, RuntimeError, ImportError)


def _sources(*relative: str) -> list[str]:
    return [str(_this_dir / path) for path in relative]


def _load_extension(name: str, sources: Sequence[str]):
    """Try to build/load an extension, honoring Ninja if available.

    - Defaults to USE_NINJA=1 (honor Ninja) but allows override via env.
    - Build logs can be enabled with VITERBO_CPP_VERBOSE=1.
    - Falls back silently on known build/import failures.
    """
    import os

    os.environ.setdefault("USE_NINJA", "1")
    verbose = os.getenv("VITERBO_CPP_VERBOSE", "0") in {"1", "true", "yes", "on"}
    try:
        return load(
            name=name,
            sources=list(sources),
            extra_cflags=["-O3"],
            verbose=verbose,
        )
    except _SAFE_EXCEPTIONS:
        return None


@lru_cache(maxsize=1)
def _load_add_one_extension():
    return _load_extension("viterbo_add_one_ext", _sources("add_one.cpp"))


def add_one(x: torch.Tensor) -> torch.Tensor:
    """Add one to a tensor, using C++ extension when available.

    Falls back to `x + 1` if the extension is unavailable.
    """
    ext = _load_add_one_extension()
    if ext is not None:
        return ext.add_one(x)
    return x + 1


@lru_cache(maxsize=1)
def _load_affine_extension():
    return _load_extension(
        "viterbo_affine_ext",
        _sources("affine_ops.cpp", "affine_bindings.cpp"),
    )


def affine_scale_shift(
    x: torch.Tensor,
    scale: float | int,
    shift: float | int,
) -> torch.Tensor:
    """Apply `scale * x + shift` elementwise.

    Uses the compiled extension when available, otherwise a Torch fallback.
    """
    ext = _load_affine_extension()
    if ext is not None:
        return ext.affine_scale_shift(x, float(scale), float(shift))
    return x * scale + shift


def has_add_one_extension() -> bool:
    """Return True when the compiled `add_one` extension is available."""
    return _load_add_one_extension() is not None


def has_affine_extension() -> bool:
    """Return True when the compiled affine extension is available."""
    return _load_affine_extension() is not None


def clear_extension_caches() -> None:
    """Reset cached extension modules (testing hook)."""
    _load_add_one_extension.cache_clear()
    _load_affine_extension.cache_clear()
