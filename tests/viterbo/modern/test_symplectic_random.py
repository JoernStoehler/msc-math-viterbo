"""Random symplectic matrices and invariance checks (smoke)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from viterbo.modern import polytopes, capacity, symplectic


@pytest.mark.goal_code
@pytest.mark.smoke
def test_random_symplectic_is_symplectic() -> None:
    """M^T J M == J within tolerance for sampled matrices."""
    key = jax.random.PRNGKey(0)
    dim = 4
    M = symplectic.random_symplectic_matrix(key, dim)
    J = symplectic.standard_symplectic_matrix(dim)
    lhs = M.T @ J @ M
    assert jnp.allclose(lhs, J, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_invariance_under_random_symplectic_4d() -> None:
    """c_EHZ invariant under linear symplectic transformations (4D)."""
    key = jax.random.PRNGKey(123)
    V = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0],
            [0.0, 0.0, 0.0, 0.8],
        ],
        dtype=jnp.float64,
    )
    P = polytopes.build_from_vertices(V)
    c0 = capacity.ehz_capacity_reference(P)
    M = symplectic.random_symplectic_matrix(key, 4)
    P2 = polytopes.build_from_vertices(V @ M.T)
    c1 = capacity.ehz_capacity_reference(P2)
    assert jnp.isclose(c0, c1, rtol=1e-9, atol=1e-12)

