"""Thin wrappers for SciPy linear algebra operations with NumPy interop."""

from __future__ import annotations

from typing import Any, cast

import numpy as _np
import numpy.typing as _npt
import scipy.linalg as _la  # type: ignore[reportMissingTypeStubs]


def expm(matrix: Any) -> _npt.NDArray[_np.float64]:
    """Matrix exponential via SciPy, returning a NumPy array."""
    A = _np.asarray(matrix, dtype=float)
    result = cast(
        _npt.NDArray[_np.float64],
        _la.expm(A),  # type: ignore[reportUnknownMemberType]
    )
    return _np.asarray(result, dtype=_np.float64)
