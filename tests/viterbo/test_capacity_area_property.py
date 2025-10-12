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
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from viterbo._wrapped import spatial as _spatial
from viterbo.math import volume
from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals


def _random_convex_polygon_vertices(rng: np.random.Generator, n: int) -> jnp.ndarray:
    pts = rng.normal(size=(n, 2))
    try:
        verts_idx = _spatial.convex_hull_vertices(jnp.asarray(pts, dtype=jnp.float64))
    except _spatial.QhullError:
        assume(False)  # degenerate sample; ask Hypothesis for another
        raise AssertionError  # unreachable
    ordered = pts[np.asarray(verts_idx, dtype=int)]
    return jnp.asarray(ordered, dtype=jnp.float64)


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=4, max_value=12), st.integers(min_value=0, max_value=2**31 - 1))
def test_capacity_equals_area_random_convex_polygons(n: int, seed: int) -> None:
    """Random 2D convex polygons satisfy c_EHZ = area with tight tolerance."""
    rng = np.random.default_rng(seed)
    verts = _random_convex_polygon_vertices(rng, n)
    # Compute half-space representation from vertices
    equations = _spatial.convex_hull_equations(verts)
    B = jnp.asarray(equations[:, :-1], dtype=jnp.float64)
    c = jnp.asarray(-equations[:, -1], dtype=jnp.float64)
    cap = ehz_capacity_reference_facet_normals(B, c)
    area = volume.volume_reference(verts)
    assert jnp.isclose(cap, area, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=10, deadline=None)
@given(
    st.integers(min_value=4, max_value=12),
    st.floats(min_value=-math.pi, max_value=math.pi),
    st.integers(min_value=0, max_value=2**31 - 1),
)
def test_capacity_rotation_invariance_2d(n: int, theta: float, seed: int) -> None:
    """In 2D, capacity is rotation invariant (equals area)."""
    rng = np.random.default_rng(seed)
    verts = _random_convex_polygon_vertices(rng, n)
    eq = _spatial.convex_hull_equations(verts)
    B = jnp.asarray(eq[:, :-1], dtype=jnp.float64)
    c = jnp.asarray(-eq[:, -1], dtype=jnp.float64)
    c0 = ehz_capacity_reference_facet_normals(B, c)

    R = jnp.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=jnp.float64
    )
    verts_rot = verts @ R.T
    eq_rot = _spatial.convex_hull_equations(verts_rot)
    B_rot = jnp.asarray(eq_rot[:, :-1], dtype=jnp.float64)
    c_rot = jnp.asarray(-eq_rot[:, -1], dtype=jnp.float64)
    c1 = ehz_capacity_reference_facet_normals(B_rot, c_rot)
    assert jnp.isclose(c0, c1, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_code
@pytest.mark.smoke
@settings(max_examples=10, deadline=None)
@given(st.integers(min_value=4, max_value=12), st.integers(min_value=0, max_value=2**31 - 1))
def test_roundtrip_vertices_to_halfspaces_back_to_vertices(n: int, seed: int) -> None:
    """V→H→V round-trip preserves hull vertex set (order up to permutation)."""
    rng = np.random.default_rng(seed)
    verts = _random_convex_polygon_vertices(rng, n)
    eq = _spatial.convex_hull_equations(verts)
    B = jnp.asarray(eq[:, :-1], dtype=jnp.float64)
    c = jnp.asarray(-eq[:, -1], dtype=jnp.float64)
    got = jnp.asarray(_spatial.halfspace_intersection_vertices(B, c), dtype=jnp.float64)
    exp = jnp.asarray(verts, dtype=jnp.float64)
    assume(got.shape[0] == exp.shape[0])
    idx_g = jnp.lexsort((got[:, 1], got[:, 0]))
    idx_e = jnp.lexsort((exp[:, 1], exp[:, 0]))
    assert got.shape == exp.shape
    assert jnp.allclose(got[idx_g], exp[idx_e], rtol=1e-9, atol=1e-12)
