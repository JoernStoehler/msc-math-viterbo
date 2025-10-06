"""Curated half-space datasets for regression tests and examples."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def unit_hypercube_halfspaces(
    dimension: int,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Return ``Bx ≤ c`` for the axis-aligned unit hypercube centered at the origin."""
    if dimension <= 0:
        msg = "dimension must be positive"
        raise ValueError(msg)

    identity = jnp.eye(dimension, dtype=jnp.float64)
    matrix = jnp.vstack((identity, -identity))
    offsets = jnp.ones(2 * dimension, dtype=jnp.float64)
    return matrix, offsets


def unit_square_halfspaces() -> tuple[
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]:
    """Return a cached 2D unit square description for convenience."""
    return unit_hypercube_halfspaces(2)
