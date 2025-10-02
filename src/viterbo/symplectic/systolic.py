"""Helper to evaluate the polytope systolic ratio."""

from __future__ import annotations

import math
from typing import Final, overload

import numpy as np
from jaxtyping import Float

from viterbo.geometry.polytopes import Polytope
from viterbo.geometry.volume import polytope_volume_fast

from .capacity import compute_ehz_capacity
from .capacity_fast import compute_ehz_capacity_fast

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_FACET_MATRIX_AXES: Final[str] = f"{_FACET_AXIS} {_DIMENSION_AXIS}"


@overload
def systolic_ratio(polytope: Polytope, /) -> float: ...


@overload
def systolic_ratio(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
    /,
) -> float: ...


def systolic_ratio(
    arg: Polytope | Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS] | None = None,
) -> float:
    """Return ``sys(K) = c_EHZ(K)^n / (n! vol_{2n}(K))`` for a ``2n``-polytope."""
    if isinstance(arg, Polytope):
        B, offsets = arg.halfspace_data()
    else:
        if c is None:
            msg = "Both B and c must be supplied for raw half-space input."
            raise ValueError(msg)
        B = np.asarray(arg, dtype=float)
        offsets = np.asarray(c, dtype=float)

    if B.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1:
        msg = "Facet offsets c must be one-dimensional."
        raise ValueError(msg)

    if B.shape[0] != offsets.shape[0]:
        msg = "Number of offsets must match the number of facets."
        raise ValueError(msg)

    dimension = B.shape[1]
    if dimension % 2 != 0:
        msg = "Systolic ratio is defined for even-dimensional symplectic spaces."
        raise ValueError(msg)

    n = dimension // 2
    try:
        capacity = compute_ehz_capacity_fast(B, offsets)
    except ValueError:
        capacity = compute_ehz_capacity(B, offsets)
    volume = polytope_volume_fast(B, offsets)
    denominator = math.factorial(n) * volume
    if denominator <= 0:
        msg = "Volume must be positive."
        raise ValueError(msg)
    return float((capacity**n) / denominator)


__all__ = ["systolic_ratio"]
