"""Builder semantics for modern polytopes (H-rep and V-rep)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import polytopes


@pytest.mark.goal_code
@pytest.mark.smoke
def test_build_from_halfspaces_enumerates_square_vertices() -> None:
    """Half-space builder produces the 4 vertices of a square [-1,1]^2."""
    normals = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    offsets = jnp.ones((4,), dtype=jnp.float64)
    P = polytopes.build_from_halfspaces(normals, offsets)
    expected = jnp.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    # Compare sets by sorting rows lexicographically
    got = jnp.asarray(P.vertices)
    assert got.shape == (4, 2)
    idx_g = jnp.lexsort((got[:, 1], got[:, 0]))
    idx_e = jnp.lexsort((expected[:, 1], expected[:, 0]))
    assert jnp.allclose(got[idx_g], expected[idx_e], rtol=1e-12, atol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_offsets_match_support_function_over_vertices() -> None:
    """For V-rep builder, offsets equal max dot(normal, vertex) across vertices."""
    verts = jnp.array(
        [
            [2.0, 1.0],
            [2.0, -1.0],
            [-2.0, 1.0],
            [-2.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    P = polytopes.build_from_vertices(verts)
    # For each facet normal, offset equals support function at that direction.
    proj = P.vertices @ P.normals.T  # shape (nv, nf)
    h = jnp.max(proj, axis=0)
    assert jnp.allclose(h, P.offsets, rtol=1e-9, atol=1e-12)

