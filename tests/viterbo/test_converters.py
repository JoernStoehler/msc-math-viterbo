"""Atlas conversion helpers for rows to modern types."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.datasets import atlas


@pytest.mark.goal_code
@pytest.mark.smoke
def test_as_polytope_and_as_cycle_shapes_and_dtypes() -> None:
    """as_polytope/as_cycle return JAX arrays with float64 points and boolean incidence."""

    dimension = 2
    normals = [
        jnp.array([1.0, 0.0]),
        jnp.array([-1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([0.0, -1.0]),
    ]
    offsets = [1.0, 1.0, 1.0, 1.0]
    vertices = [
        jnp.array([1.0, 1.0]),
        jnp.array([1.0, -1.0]),
        jnp.array([-1.0, 1.0]),
        jnp.array([-1.0, -1.0]),
    ]

    poly = atlas.as_polytope(
        dimension=dimension,
        num_facets=len(normals),
        num_vertices=len(vertices),
        normals=normals,
        offsets=offsets,
        vertices=vertices,
    )
    assert poly.normals.dtype == jnp.float64
    assert poly.vertices.dtype == jnp.float64
    assert poly.offsets.dtype == jnp.float64
    assert poly.incidence.dtype == jnp.bool_

    points = [jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])]
    cyc = atlas.as_cycle(
        dimension=dimension,
        num_points=len(points),
        points=points,
        polytope=poly,
    )
    assert cyc.points.dtype == jnp.float64
    assert cyc.incidence.dtype == jnp.bool_
