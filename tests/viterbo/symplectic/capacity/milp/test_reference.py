"""Tests for the MILP reference solver."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from viterbo.geometry.polytopes import (
    Polytope,
    simplex_with_uniform_weights,
    truncated_simplex_four_dim,
)
from viterbo.symplectic.capacity import compute_ehz_capacity_reference
from viterbo.symplectic.capacity.milp.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference_milp,
)


@pytest.fixture(scope="module")
def simplex_polytope() -> Polytope:
    return simplex_with_uniform_weights(4, name="milp-simplex")


@pytest.fixture(scope="module")
def truncated_simplex() -> Polytope:
    return truncated_simplex_four_dim()


@pytest.mark.parametrize("polytope_fixture", ["simplex_polytope", "truncated_simplex"])
def test_reference_matches_facet_solution(
    polytope_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Reference MILP solver reproduces the facet-normal optimum."""

    polytope: Polytope = request.getfixturevalue(polytope_fixture)
    B, c = polytope.halfspace_data()

    facet_value = compute_ehz_capacity_reference(B, c)
    result = compute_ehz_capacity_reference_milp(B, c)

    assert math.isclose(result.upper_bound, facet_value, rel_tol=0.0, abs_tol=1e-9)
    assert result.lower_bound is not None
    assert 0.0 <= result.lower_bound <= result.upper_bound
    assert result.gap_ratio is not None
    assert 0.0 <= result.gap_ratio <= 1.0

    certificate = result.certificate
    beta = jnp.asarray(certificate.beta)
    normals = jnp.asarray(B)[jnp.asarray(certificate.subset_indices), :]
    support = jnp.asarray(c)[jnp.asarray(certificate.subset_indices)]

    # Verify Reeb-measure feasibility from the certificate.
    assert bool(jnp.all(beta >= -1e-12))
    residual = normals.T @ beta
    assert bool(jnp.all(jnp.isclose(residual, jnp.zeros_like(residual), atol=1e-9, rtol=0.0)))
    assert math.isclose(float(support @ beta), 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_reference_explores_all_subsets(simplex_polytope: Polytope) -> None:
    """Reference MILP solver enumerates all subsets of size d+1."""

    B, c = simplex_polytope.halfspace_data()
    num_facets = int(B.shape[0])
    subset_size = int(B.shape[1]) + 1
    expected = math.comb(num_facets, subset_size)

    result = compute_ehz_capacity_reference_milp(B, c)
    assert result.explored_subsets == expected
