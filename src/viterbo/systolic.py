"""Systolic ratio helpers wired to the modern capacity implementations."""

from __future__ import annotations

import math
from typing import overload

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geom import Polytope as GeometryPolytope
from viterbo.geom import polytope_volume_fast
from viterbo.capacity import ehz_capacity_fast, ehz_capacity_reference
from viterbo.types import Polytope


@overload
def systolic_ratio(polytope: Polytope | GeometryPolytope, /) -> float: ...


@overload
def systolic_ratio(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    /,
) -> float: ...


def systolic_ratio(
    arg: Polytope | GeometryPolytope | Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"] | None = None,
) -> float:
    """Return ``sys(K) = c_EHZ(K)^n / (n! vol_{2n}(K))`` for a ``2n``-polytope."""

    if isinstance(arg, Polytope):
        B_matrix = jnp.asarray(arg.normals, dtype=jnp.float64)
        offsets = jnp.asarray(arg.offsets, dtype=jnp.float64)
    elif isinstance(arg, GeometryPolytope):
        B_matrix, offsets = arg.halfspace_data()
        B_matrix = jnp.asarray(B_matrix, dtype=jnp.float64)
        offsets = jnp.asarray(offsets, dtype=jnp.float64)
    else:
        if c is None:
            msg = "Both B and c must be supplied for raw half-space input."
            raise ValueError(msg)
        B_matrix = jnp.asarray(arg, dtype=jnp.float64)
        offsets = jnp.asarray(c, dtype=jnp.float64)

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
        capacity = ehz_capacity_fast(
            Polytope(
                normals=B_matrix,
                offsets=offsets,
                vertices=jnp.empty((0, dimension), dtype=jnp.float64),
                incidence=jnp.empty((0, B_matrix.shape[0]), dtype=bool),
            )
        )
    except ValueError:
        capacity = ehz_capacity_reference(
            Polytope(
                normals=B_matrix,
                offsets=offsets,
                vertices=jnp.empty((0, dimension), dtype=jnp.float64),
                incidence=jnp.empty((0, B_matrix.shape[0]), dtype=bool),
            )
        )

    volume = polytope_volume_fast(B_matrix, offsets)
    denominator = math.factorial(n) * volume
    if denominator <= 0:
        msg = "Volume must be positive."
        raise ValueError(msg)

    return float((capacity**n) / denominator)


__all__ = ["systolic_ratio"]
