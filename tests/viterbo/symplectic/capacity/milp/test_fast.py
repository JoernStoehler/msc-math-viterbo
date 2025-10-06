"""Tests for the heuristic MILP solver."""

from __future__ import annotations

import math

import pytest

from viterbo.geometry.polytopes import (
    Polytope,
    simplex_with_uniform_weights,
    truncated_simplex_four_dim,
)
from viterbo.symplectic.capacity import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast_facet,
    compute_ehz_capacity_reference,
)
from viterbo.symplectic.capacity.milp.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast_milp,
)


@pytest.fixture(scope="module")
def four_dimensional_simplex() -> Polytope:
    return simplex_with_uniform_weights(4, name="milp-fast-simplex")


@pytest.fixture(scope="module")
def truncated_simplex() -> Polytope:
    return truncated_simplex_four_dim()


@pytest.mark.parametrize("polytope_fixture", ["four_dimensional_simplex", "truncated_simplex"])
def test_fast_solver_matches_reference(
    polytope_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Heuristic solver finds the same certificate as the facet reference on small cases."""

    polytope: Polytope = request.getfixturevalue(polytope_fixture)
    B, c = polytope.halfspace_data()

    reference_value = compute_ehz_capacity_reference(B, c)
    fast_result = compute_ehz_capacity_fast_milp(B, c, node_limit=4096)

    assert math.isclose(fast_result.upper_bound, reference_value, rel_tol=0.0, abs_tol=1e-9)
    assert fast_result.lower_bound is None
    assert fast_result.explored_subsets >= 1


def test_fast_solver_improves_over_facet_fast(four_dimensional_simplex: Polytope) -> None:
    """MILP heuristic matches or improves the JAX-first fast solver's action."""

    B, c = four_dimensional_simplex.halfspace_data()
    facet_value = compute_ehz_capacity_fast_facet(B, c)
    milp_value = compute_ehz_capacity_fast_milp(B, c, node_limit=1024).upper_bound

    assert math.isclose(milp_value, facet_value, rel_tol=0.0, abs_tol=1e-9)
