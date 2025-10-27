"""C++ extension scaffold via torch.utils.cpp_extension.

Provides lazy loaders for example ops implemented in C++.
Build/import failures raise a standard error with tooling diagnostics.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

_this_dir = Path(__file__).parent
_SAFE_EXCEPTIONS = (OSError, RuntimeError, ImportError, subprocess.CalledProcessError)


def _sources(*relative: str) -> list[str]:
    return [str(_this_dir / path) for path in relative]


def _load_extension(name: str, sources: Sequence[str]) -> Any:
    """Build/load an extension, honoring Ninja if available.

    - Defaults to USE_NINJA=1 (honor Ninja) but allows override via env.
    - Build logs can be enabled with VITERBO_CPP_VERBOSE=1.
    - Raises ImportError with concise diagnostics on failure.
    """
    import os

    os.environ.setdefault("USE_NINJA", "1")
    if "CXX" not in os.environ and "CC" in os.environ:
        # Torch uses CXX for C++ builds; mirror CC overrides when CXX is unset.
        os.environ["CXX"] = os.environ["CC"]
    verbose = os.getenv("VITERBO_CPP_VERBOSE", "0") in {"1", "true", "yes", "on"}
    try:
        return load(
            name=name,
            sources=list(sources),
            extra_cflags=["-O3"],
            verbose=verbose,
        )
    except _SAFE_EXCEPTIONS as exc:
        raise ImportError(
            f"Failed to build or load the '{name}' C++ extension. "
            "Verify that a compatible C++ compiler and Ninja are installed "
            "(override with USE_NINJA=0 if Ninja is unavailable). "
            "Re-run with VITERBO_CPP_VERBOSE=1 for verbose build logs."
        ) from exc


@lru_cache(maxsize=1)
def _load_add_one_extension() -> Any:
    return _load_extension("viterbo_add_one_ext", _sources("add_one.cpp"))


def add_one(x: torch.Tensor) -> torch.Tensor:
    """Add one to a tensor via the compiled extension.

    Raises ImportError if the extension cannot be built or loaded.
    """
    return _load_add_one_extension().add_one(x)


@lru_cache(maxsize=1)
def _load_affine_extension() -> Any:
    return _load_extension(
        "viterbo_affine_ext",
        _sources("affine_ops.cpp", "affine_bindings.cpp"),
    )


def affine_scale_shift(
    x: torch.Tensor,
    scale: float | int,
    shift: float | int,
) -> torch.Tensor:
    """Apply `scale * x + shift` elementwise via the compiled extension.

    Raises ImportError if the extension cannot be built or loaded.
    """
    ext = _load_affine_extension()
    return ext.affine_scale_shift(x, float(scale), float(shift))


def has_add_one_extension() -> bool:
    """Return True when the compiled `add_one` extension is available."""
    try:
        _load_add_one_extension()
    except ImportError:
        return False
    return True


def has_affine_extension() -> bool:
    """Return True when the compiled affine extension is available."""
    try:
        _load_affine_extension()
    except ImportError:
        return False
    return True


def clear_extension_caches() -> None:
    """Reset cached extension modules (testing hook)."""
    _load_add_one_extension.cache_clear()
    _load_affine_extension.cache_clear()


# Eagerly load extensions on import to surface configuration issues early.
_load_add_one_extension()
_load_affine_extension()
