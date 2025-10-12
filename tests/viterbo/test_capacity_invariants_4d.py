"""4D capacity invariants for the modern API.

Smoke-level tests focused on basic invariants for a simple 4D simplex:
- Quadratic scaling under linear scaling
- Monotonicity under inclusion
- Invariance under block-diagonal symplectic rotations
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from viterbo import capacity, polytopes


def _simplex_4d(edge: float = 2.0) -> jnp.ndarray:
    """Return vertices of conv{0, edge*e1, edge*e2, edge*e3, edge*e4}."""
    e = edge
    return jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [e, 0.0, 0.0, 0.0],
            [0.0, e, 0.0, 0.0],
            [0.0, 0.0, e, 0.0],
            [0.0, 0.0, 0.0, e],
        ],
        dtype=jnp.float64,
    )


def _block_rot(theta: float) -> jnp.ndarray:
    """Symplectic, orthogonal 4x4 block-diagonal rotation diag(R_theta, R_theta)."""
    c = math.cos(theta)
    s = math.sin(theta)
    R = jnp.asarray([[c, -s], [s, c]], dtype=jnp.float64)
    Z = jnp.zeros_like(R)
    return jnp.block([[R, Z], [Z, R]])


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_scales_quadratically_in_4d_on_simplex() -> None:
    """c_EHZ(sP) == s^2 c_EHZ(P) on a 4D simplex."""
    s = 1.5
    V = _simplex_4d(edge=2.0)
    P = polytopes.build_from_vertices(V)
    c0 = capacity.ehz_capacity_reference(P)
    P_s = polytopes.build_from_vertices(V * s)
    c1 = capacity.ehz_capacity_reference(P_s)
    assert jnp.isclose(c1, (s**2) * c0, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_monotone_under_inclusion_4d_simplex() -> None:
    """If P ⊂ Q then c_EHZ(P) <= c_EHZ(Q) on 4D simplex family by scaling."""
    V = _simplex_4d(edge=2.0)
    P_small = polytopes.build_from_vertices(V)
    P_large = polytopes.build_from_vertices(2.0 * V)
    c_small = capacity.ehz_capacity_reference(P_small)
    c_large = capacity.ehz_capacity_reference(P_large)
    assert float(c_large) >= float(c_small)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_symplectic_invariance_block_rotation_4d() -> None:
    """Capacity invariant under diag(R_theta, R_theta) rotations in R^4."""
    V = _simplex_4d(edge=2.0)
    P = polytopes.build_from_vertices(V)
    c0 = capacity.ehz_capacity_reference(P)
    M = _block_rot(theta=1.234)
    V_rot = V @ M.T
    P_rot = polytopes.build_from_vertices(V_rot)
    c1 = capacity.ehz_capacity_reference(P_rot)
    assert jnp.isclose(c0, c1, rtol=1e-9, atol=1e-12)
