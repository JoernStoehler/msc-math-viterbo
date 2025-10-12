"""Cycle extraction interfaces and expected output shapes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import atlas, cycles


@pytest.mark.goal_code
@pytest.mark.smoke
def test_minimum_cycle_reference_returns_closed_cycle_4d() -> None:
    """Reference cycle returns a set of 4D points on the boundary (4D only)."""
    # Product square [-1,1]^2 x [-1,1]^2 in R^4
    normals = [
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jnp.array([-1.0, 0.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0, 0.0]),
        jnp.array([0.0, -1.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0, 0.0]),
        jnp.array([0.0, 0.0, -1.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0, 1.0]),
        jnp.array([0.0, 0.0, 0.0, -1.0]),
    ]
    offsets = [1.0] * 8
    # 16 vertices with coordinates in {±1}^4
    vertices = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                for w in (-1.0, 1.0):
                    vertices.append(jnp.array([x, y, z, w], dtype=jnp.float64))
    bundle = atlas.as_polytope(4, len(normals), len(vertices), normals, offsets, vertices)
    cyc = cycles.minimum_cycle_reference(bundle)
    assert cyc.shape[1] == 4


@pytest.mark.goal_code
@pytest.mark.smoke
def test_minimum_cycle_batched_returns_padded_cycles() -> None:
    """Batched cycle extraction returns padded cycles with batch dimension first."""
    normals = jnp.zeros((2, 3, 4), dtype=jnp.float64)
    offsets = jnp.zeros((2, 3), dtype=jnp.float64)
    padded_vertices = jnp.zeros((2, 5, 4), dtype=jnp.float64)
    cyc = cycles.minimum_cycle_batched(normals, offsets, padded_vertices=padded_vertices)
    assert cyc.shape[0] == 2
