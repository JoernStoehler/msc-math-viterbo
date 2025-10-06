"""Reference solver regression tests for combinatorial Reeb cycles."""

from __future__ import annotations

import math

import pytest

from viterbo.geometry.polytopes import (
    simplex_with_uniform_weights,
    truncated_simplex_four_dim,
)
from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as facet_reference,
)
from viterbo.symplectic.capacity.reeb_cycles.reference import (
    compute_ehz_capacity_reference,
)


@pytest.mark.parametrize(
    "polytope_factory",
    [
        truncated_simplex_four_dim,
        lambda: simplex_with_uniform_weights(4, name="uniform-simplex-4d"),
    ],
)
def test_reference_matches_facet_normals(polytope_factory) -> None:
    polytope = polytope_factory()
    B, c = polytope.halfspace_data()
    reeb_capacity = compute_ehz_capacity_reference(B, c)
    facet_capacity = facet_reference(B, c)
    assert math.isclose(reeb_capacity, facet_capacity, rel_tol=0.0, abs_tol=1e-8)


def test_reference_agrees_with_polytope_metadata() -> None:
    polytope = truncated_simplex_four_dim()
    assert polytope.reference_capacity is not None
    B, c = polytope.halfspace_data()
    reeb_capacity = compute_ehz_capacity_reference(B, c)
    assert math.isclose(reeb_capacity, polytope.reference_capacity, rel_tol=0.0, abs_tol=1e-9)
