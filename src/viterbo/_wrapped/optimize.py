"""Thin wrapper for SciPy optimize linprog with explicit JAX↔NumPy conversion."""

from __future__ import annotations

import importlib
from typing import Any, Sequence, Tuple, cast

import numpy as _np

BoundTuple = Tuple[float | None, float | None]


class _OptimizeResultProtocol:
    x: _np.ndarray | None
    fun: float | None
    success: bool
    status: int | str
    message: str | None


def linprog(
    *,
    c: Any,
    A_ub: Any | None,
    b_ub: Any | None,
    A_eq: Any | None,
    b_eq: Any | None,
    bounds: Sequence[BoundTuple] | None,
    **options: Any,
) -> _OptimizeResultProtocol:
    """Call SciPy's linprog converting all arrays to NumPy internally."""
    scipy_optimize = importlib.import_module("scipy.optimize")
    func: Any = getattr(scipy_optimize, "linprog")
    return cast(
        _OptimizeResultProtocol,
        func(
            c=_np.asarray(c, dtype=float),
            A_ub=None if A_ub is None else _np.asarray(A_ub, dtype=float),
            b_ub=None if b_ub is None else _np.asarray(b_ub, dtype=float),
            A_eq=None if A_eq is None else _np.asarray(A_eq, dtype=float),
            b_eq=None if b_eq is None else _np.asarray(b_eq, dtype=float),
            bounds=bounds,
            **options,
        ),
    )
