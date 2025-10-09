from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.examples import hypercube
from viterbo.exp1.polytopes import (
    HalfspacePolytope,
    lagrangian_product,
    to_halfspaces,
    to_lagrangian_product,
    to_vertices,
)


@pytest.mark.goal_code
def test_halfspace_as_tuple_roundtrip() -> None:
    """as_tuple returns (A,b) matching construction arrays and shapes."""
    cube = hypercube(2)
    A, b = cube.as_tuple()
    P = HalfspacePolytope(normals=A, offsets=b)
    A2, b2 = P.as_tuple()
    assert A2.shape == A.shape and b2.shape == b.shape
    assert jnp.allclose(A2, A) and jnp.allclose(b2, b)


@pytest.mark.goal_code
def test_lagrangian_product_construction_and_factorization() -> None:
    """lagrangian_product builds a valid 2x2 product and factorization succeeds."""
    left = to_vertices(hypercube(2))
    right = to_vertices(hypercube(2))
    prod = lagrangian_product(left, right)
    # Convert back to 4D halfspaces and factor
    H = to_halfspaces(prod)
    factored = to_lagrangian_product(H)
    # Ensure the block counts match and vertices are non-empty
    assert factored.verts_p.shape[1] == 2 and factored.verts_q.shape[1] == 2
    assert factored.verts_p.shape[0] > 0 and factored.verts_q.shape[0] > 0
