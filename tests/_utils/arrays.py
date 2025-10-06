"""Array helpers for tests (assertions and conversions)."""

from __future__ import annotations

import numpy as np


def as_np(x: object) -> np.ndarray:
    """Convert JAX/NumPy arrays to a NumPy ndarray with float dtype."""
    return np.asarray(x, dtype=float)


def assert_allclose_arr(
    actual: object, expected: object, *, rtol: float = 1e-9, atol: float = 0.0
) -> None:
    """Assert two arrays are close, accepting JAX or NumPy arrays."""
    np.testing.assert_allclose(as_np(actual), as_np(expected), rtol=rtol, atol=atol)


def sorted_rows(x: object) -> np.ndarray:
    """Return rows sorted lexicographically as a NumPy array."""
    arr = as_np(x)
    keys = np.lexsort(arr.T[::-1])
    return arr[keys]
