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
from viterbo.symplectic.capacity.facet_normals.subset_utils import FacetSubset
from viterbo.symplectic.capacity.milp.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference_milp,
)


@pytest.fixture(scope="module")
def simplex_polytope() -> Polytope:
    return simplex_with_uniform_weights(4, name="milp-simplex")


@pytest.fixture(scope="module")
def truncated_simplex() -> Polytope:
    return truncated_simplex_four_dim()


@pytest.mark.goal_math
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
    assert result.gap_absolute is not None
    assert 0.0 <= result.gap_absolute <= result.upper_bound

    certificate = result.certificate
    assert isinstance(certificate.subset, FacetSubset)
    assert math.isclose(certificate.capacity, result.upper_bound, rel_tol=0.0, abs_tol=1e-12)

    subset = certificate.subset
    beta = jnp.asarray(subset.beta)
    normals = jnp.asarray(B)[jnp.asarray(subset.indices), :]
    support = jnp.asarray(c)[jnp.asarray(subset.indices)]

    # Verify Reeb-measure feasibility from the certificate.
    assert bool(jnp.all(beta >= -1e-12))
    residual = normals.T @ beta
    assert bool(jnp.all(jnp.isclose(residual, jnp.zeros_like(residual), atol=1e-9, rtol=0.0)))
    assert math.isclose(float(support @ beta), 1.0, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.goal_code
def test_reference_explores_all_subsets(simplex_polytope: Polytope) -> None:
    """Reference MILP solver enumerates all d+1-sized facet subsets."""

    B, c = simplex_polytope.halfspace_data()
    num_facets = int(B.shape[0])
    subset_size = int(B.shape[1]) + 1
    expected = math.comb(num_facets, subset_size)

    result = compute_ehz_capacity_reference_milp(B, c)
    assert result.explored_subsets == expected
