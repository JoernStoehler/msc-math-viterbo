"""Optimized combinatorial Reeb cycle solver tests."""

from __future__ import annotations

import math

import pytest

from viterbo.geometry.polytopes import (
    simplex_with_uniform_weights,
    truncated_simplex_four_dim,
)
from viterbo.symplectic.capacity.reeb_cycles.fast import (
    compute_ehz_capacity_fast,
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
def test_fast_matches_reference(polytope_factory) -> None:
    polytope = polytope_factory()
    B, c = polytope.halfspace_data()
    fast_value = compute_ehz_capacity_fast(B, c)
    reference_value = compute_ehz_capacity_reference(B, c)
    assert math.isclose(fast_value, reference_value, rel_tol=0.0, abs_tol=1e-8)
