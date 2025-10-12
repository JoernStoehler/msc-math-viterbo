"""Systolic ratio helpers wired to the modern capacity implementations."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.math.volume import polytope_volume_fast
from viterbo.math.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
)


def systolic_ratio(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
) -> float:
    """Return ``sys(K) = c_EHZ(K)^n / (n! vol_{2n}(K))`` for a ``2n``-polytope."""

    B_matrix = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)

    if B_matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1:
        msg = "Facet offsets c must be one-dimensional."
        raise ValueError(msg)

    if B_matrix.shape[0] != offsets.shape[0]:
        msg = "Number of offsets must match the number of facets."
        raise ValueError(msg)

    dimension = int(B_matrix.shape[1])
    if dimension % 2 != 0:
        msg = "Systolic ratio is defined for even-dimensional symplectic spaces."
        raise ValueError(msg)

    n = dimension // 2
    try:
        capacity = ehz_capacity_fast_facet_normals(B_matrix, offsets)
    except ValueError:
        capacity = ehz_capacity_reference_facet_normals(B_matrix, offsets)

    volume = polytope_volume_fast(B_matrix, offsets)
    denominator = math.factorial(n) * volume
    if denominator <= 0:
        msg = "Volume must be positive."
        raise ValueError(msg)

    return float((capacity**n) / denominator)


__all__ = ["systolic_ratio"]
