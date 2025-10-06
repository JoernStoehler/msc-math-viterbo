"""Shared kernels for volume estimators."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float


def volume_of_simplices(
    simplex_vertices: Float[Array, " num_simplices vertices dimension"],
) -> float:
    """Return the total volume of stacked simplices."""
    v = jnp.asarray(simplex_vertices)
    base = v[:, 0, :]
    edges = v[:, 1:, :] - base[:, None, :]
    determinants = jnp.linalg.det(edges)
    dimension = v.shape[2]
    total = jnp.sum(jnp.abs(determinants)) / math.factorial(int(dimension))
    return float(total)
