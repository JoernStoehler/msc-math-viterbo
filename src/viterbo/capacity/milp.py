"""MILP-inspired capacity envelopes built on modern primitives."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.capacity import facet_normals
from viterbo.types import MilpCapacityBounds

def _capacity_upper_bound(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
) -> float:
    normals = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    return facet_normals.ehz_capacity_reference_facet_normals(normals, offsets)


def ehz_capacity_reference_milp(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    max_nodes: int = 1024,
) -> MilpCapacityBounds:
    """Return a deterministic upper bound inspired by exhaustive MILP search."""
    upper = _capacity_upper_bound(normals, offsets)
    return (0.0, upper, max_nodes, "exhaustive")


def ehz_capacity_fast_milp(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    node_limit: int = 256,
) -> MilpCapacityBounds:
    """Return a lightweight MILP-style certificate based on support radii."""
    try:
        upper = float(facet_normals.ehz_capacity_fast_facet_normals(normals, offsets))
    except ValueError:
        upper = float(facet_normals.ehz_capacity_reference_facet_normals(normals, offsets))
    return (0.0, upper, node_limit, "heuristic")


__all__ = [
    "ehz_capacity_reference_milp",
    "ehz_capacity_fast_milp",
]
