"""Dataset-level adapters for systolic ratio computation."""

from __future__ import annotations

from typing import overload

import jax.numpy as jnp

from viterbo.datasets.types import HalfspaceData, Polytope, PolytopeRecord
from viterbo.math.systolic import systolic_ratio as systolic_ratio_halfspaces


@overload
def systolic_ratio(polytope: Polytope | PolytopeRecord, /) -> float: ...


@overload
def systolic_ratio(geometry: HalfspaceData, /) -> float: ...


def systolic_ratio(arg: Polytope | PolytopeRecord | HalfspaceData, /) -> float:
    """Compute the systolic ratio for various dataset geometry containers."""

    if isinstance(arg, PolytopeRecord):
        normals = jnp.asarray(arg.geometry.normals, dtype=jnp.float64)
        offsets = jnp.asarray(arg.geometry.offsets, dtype=jnp.float64)
    elif isinstance(arg, Polytope):
        normals = jnp.asarray(arg.normals, dtype=jnp.float64)
        offsets = jnp.asarray(arg.offsets, dtype=jnp.float64)
    else:
        normals, offsets = arg
        normals = jnp.asarray(normals, dtype=jnp.float64)
        offsets = jnp.asarray(offsets, dtype=jnp.float64)

    return float(systolic_ratio_halfspaces(normals, offsets))


__all__ = ["systolic_ratio"]
