"""Incidence matrix semantics for modern polytopes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest


@pytest.mark.goal_math
@pytest.mark.smoke
def test_incidence_matrix_for_axis_aligned_square() -> None:
    """For a square [-1,1]^2, each vertex lies on exactly two facets."""

    # Square in R^2: constraints
    # x <= 1, -x <= 1, y <= 1, -y <= 1
    normals = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    offsets = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
    vertices = jnp.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=jnp.float64,
    )

    # Incidence: dot(normals, vertex) == offset
    M = jnp.isclose(vertices @ normals.T, offsets[None, :], rtol=1e-9, atol=1e-12)
    # Booleans, shape (4 vertices, 4 facets)
    assert M.shape == (4, 4)
    assert M.dtype == jnp.bool_
    # Each vertex incident to exactly two facets
    assert jnp.all(M.sum(axis=1) == 2)
