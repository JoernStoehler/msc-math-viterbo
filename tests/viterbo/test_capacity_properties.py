"""Capacity invariants and properties for the modern API.

Covers:
- 2D identity: capacity equals area (volume_reference).
- Scaling in 2D: capacity scales quadratically with linear scaling.
- 4D product: returns a finite, nonnegative scalar on a simple instance.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals
from viterbo.datasets.builders import build_from_vertices


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_equals_area_in_2d_for_rectangles() -> None:
    """For axis-aligned rectangles, c_EHZ equals area (2D identity)."""
    # Rectangle [−a,a]×[−b,b]
    a, b = 2.0, 3.0
    vertices = jnp.array(
        [
            [a, b],
            [a, -b],
            [-a, b],
            [-a, -b],
        ],
        dtype=jnp.float64,
    )
    P = build_from_vertices(vertices)
    c = ehz_capacity_reference_facet_normals(P.normals, P.offsets)
    area = 4.0 * a * b  # side lengths are 2a and 2b → area = 4ab
    assert jnp.isclose(c, area, rtol=1e-12, atol=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_scales_quadratically_in_2d() -> None:
    """In 2D, capacity scales as s^2 under x -> s x."""
    s = 1.7
    base = jnp.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    P = build_from_vertices(base)
    c0 = ehz_capacity_reference_facet_normals(P.normals, P.offsets)
    P_scaled = build_from_vertices(base * s)
    c1 = ehz_capacity_reference_facet_normals(P_scaled.normals, P_scaled.offsets)
    assert jnp.isclose(c1, (s**2) * c0, rtol=1e-12, atol=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_nonnegative_on_4d_simplex() -> None:
    """Facet-normal reference returns a finite, nonnegative value for a 4D simplex."""
    # Simplex: conv{0, 2e1, 2e2, 2e3, 2e4}
    verts = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    P = build_from_vertices(verts)
    c = ehz_capacity_reference_facet_normals(P.normals, P.offsets)
    assert math.isfinite(float(c)) and float(c) >= 0.0
