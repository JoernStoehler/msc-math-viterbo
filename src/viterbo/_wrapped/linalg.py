"""Thin wrappers for SciPy linear algebra operations with NumPy interop."""

from __future__ import annotations

from typing import Any

import numpy as _np
import scipy.linalg as _la  # type: ignore[reportMissingTypeStubs]


def expm(matrix: Any) -> _np.ndarray:
    """Matrix exponential via SciPy, returning a NumPy array."""
    A = _np.asarray(matrix, dtype=float)
    return _la.expm(A)

