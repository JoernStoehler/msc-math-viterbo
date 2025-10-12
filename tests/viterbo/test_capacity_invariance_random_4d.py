"""Deep-tier invariance tests for 4D capacity on random hulls.

Covers symplectic invariance and monotonicity under inclusion on small random
4D polytopes to sanity-check algorithmic behaviour beyond curated shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from viterbo.datasets import builders as polytopes
from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals
from viterbo.math import symplectic


def _random_convex_hull_vertices_4d(key: jax.Array, n_vertices: int) -> jnp.ndarray:
    """Return hull-ordered vertices from random points in R^4 (small n)."""
    import numpy as _np
    import scipy.spatial as _spatial  # type: ignore[reportMissingTypeStubs]

    pts = jax.random.normal(key, (n_vertices, 4), dtype=jnp.float64)
    arr = _np.asarray(pts, dtype=float)
    hull = _spatial.ConvexHull(arr)
    ordered = arr[hull.vertices]
    return jnp.asarray(ordered, dtype=jnp.float64)


@pytest.mark.goal_math
@pytest.mark.deep
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_capacity_symplectic_invariance_random_hull(seed: int) -> None:
    """c_EHZ invariant under random Sp(4) transforms on random small hulls."""
    key = jax.random.PRNGKey(seed)
    V = _random_convex_hull_vertices_4d(key, n_vertices=7)
    P = polytopes.build_from_vertices(V)
    c0 = ehz_capacity_reference_facet_normals(P.normals, P.offsets)
    M = symplectic.random_symplectic_matrix(key, 4)
    V2 = V @ M.T
    P2 = polytopes.build_from_vertices(V2)
    c1 = ehz_capacity_reference_facet_normals(P2.normals, P2.offsets)
    assert jnp.isclose(c0, c1, rtol=1e-7, atol=1e-9)


@pytest.mark.goal_math
@pytest.mark.deep
@pytest.mark.parametrize("seed", [3, 4])
def test_capacity_monotonicity_random_hull(seed: int) -> None:
    """Monotonicity under inclusion via scaling on random hulls."""
    key = jax.random.PRNGKey(seed)
    V = _random_convex_hull_vertices_4d(key, n_vertices=6)
    P = polytopes.build_from_vertices(V)
    c0 = ehz_capacity_reference_facet_normals(P.normals, P.offsets)
    P_big = polytopes.build_from_vertices(1.5 * V)
    c1 = ehz_capacity_reference_facet_normals(P_big.normals, P_big.offsets)
    assert float(c1) >= float(c0)
