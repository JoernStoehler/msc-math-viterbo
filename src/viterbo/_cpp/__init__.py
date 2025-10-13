"""C++ extension scaffold via torch.utils.cpp_extension.

Provides a lazy loader for a tiny example op `add_one` implemented in C++.
If build/import fails, a pure-PyTorch fallback is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_this_dir = Path(__file__).parent


def _try_load_add_one(name: str = "add_one_ext"):
    src = _this_dir / "add_one.cpp"
    try:
        return load(name=name, sources=[str(src)], verbose=False)
    except Exception:
        return None


_ext = _try_load_add_one()


def add_one(x: torch.Tensor) -> torch.Tensor:
    """Add one to a tensor, using C++ extension when available.

    Falls back to `x + 1` if the extension is unavailable.
    """
    if _ext is not None:
        return _ext.add_one(x)
    return x + 1


__all__ = ["add_one"]

