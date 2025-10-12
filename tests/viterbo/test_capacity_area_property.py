"""Property-based tests for 2D capacity = area and invariances.

Covers fast feedback for mathematical correctness in 2D:
- Capacity equals polygon area for convex polytopes.
- Invariance under rotation (symplectic in 2D).
- V→H→V round-trip preserves vertex set (up to permutation).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from viterbo import capacity, polytopes, volume


def _random_convex_polygon_vertices(rng: np.random.Generator, n: int) -> jnp.ndarray:
    pts = rng.normal(size=(n, 2))
    # Use numpy hull to order vertices for a simple polygon; fall back to points
    import scipy.spatial as spatial  # type: ignore[reportMissingTypeStubs]

    hull = spatial.ConvexHull(pts)
    ordered = pts[hull.vertices]
    return jnp.asarray(ordered, dtype=jnp.float64)


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=15, deadline=None)
@given(st.integers(min_value=4, max_value=12), st.integers(min_value=0, max_value=2**31 - 1))
def test_capacity_equals_area_random_convex_polygons(n: int, seed: int) -> None:
    """Random 2D convex polygons satisfy c_EHZ = area with tight tolerance."""
    rng = np.random.default_rng(seed)
    verts = _random_convex_polygon_vertices(rng, n)
    P = polytopes.build_from_vertices(verts)
    c = capacity.ehz_capacity_reference(P.normals, P.offsets, P.vertices)
    a = volume.volume_reference(P)
    assert jnp.isclose(c, a, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=15, deadline=None)
@given(st.integers(min_value=4, max_value=12), st.floats(min_value=-np.pi, max_value=np.pi), st.integers(min_value=0, max_value=2**31 - 1))
def test_capacity_rotation_invariance_2d(n: int, theta: float, seed: int) -> None:
    """In 2D, capacity is rotation invariant (equals area)."""
    rng = np.random.default_rng(seed)
    verts = _random_convex_polygon_vertices(rng, n)
    P = polytopes.build_from_vertices(verts)
    c0 = capacity.ehz_capacity_reference(P.normals, P.offsets, P.vertices)
    R = jnp.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=jnp.float64)
    verts_rot = verts @ R.T
    P_rot = polytopes.build_from_vertices(verts_rot)
    c1 = capacity.ehz_capacity_reference(P_rot.normals, P_rot.offsets, P_rot.vertices)
    assert jnp.isclose(c0, c1, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_code
@pytest.mark.smoke
@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=4, max_value=10), st.integers(min_value=0, max_value=2**31 - 1))
def test_roundtrip_vertices_to_halfspaces_back_to_vertices(n: int, seed: int) -> None:
    """V→H→V round-trip preserves hull vertex set (order up to permutation)."""
    rng = np.random.default_rng(seed)
    verts = _random_convex_polygon_vertices(rng, n)
    P_v = polytopes.build_from_vertices(verts)
    P_h = polytopes.build_from_halfspaces(P_v.normals, P_v.offsets)
    got = jnp.asarray(P_h.vertices)
    exp = jnp.asarray(P_v.vertices)
    assert got.shape[1] == 2 and exp.shape[1] == 2
    # Compare sets by sorting lexicographically
    idx_g = jnp.lexsort((got[:, 1], got[:, 0]))
    idx_e = jnp.lexsort((exp[:, 1], exp[:, 0]))
    assert got.shape == exp.shape
    assert jnp.allclose(got[idx_g], exp[idx_e], rtol=1e-9, atol=1e-12)
