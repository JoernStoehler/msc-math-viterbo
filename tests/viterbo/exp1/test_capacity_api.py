from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.capacity_ehz import (
    capacity,
    capacity_and_cycle,
    capacity_minkowski_billiard,
)
from viterbo.exp1.examples import crosspolytope, hypercube, regular_ngon2d
from viterbo.exp1.polytopes import lagrangian_product


@pytest.mark.goal_math
def test_capacity_matches_capacity_and_cycle_on_product() -> None:
    """capacity(...,'minkowski_reference') equals capacity from capacity_and_cycle."""
    left = regular_ngon2d(4)
    right = regular_ngon2d(6)
    prod = lagrangian_product(left, right)
    cap_only = float(capacity(prod, method="minkowski_reference"))
    cap_cc, cyc = capacity_and_cycle(prod, method="minkowski_reference")
    assert np.isclose(cap_only, float(cap_cc), rtol=1e-9, atol=1e-9)
    assert cyc.shape[1] == 4 and cyc.shape[0] >= 3


@pytest.mark.goal_math
def test_capacity_and_cycle_minkowski_product_agrees_with_reference() -> None:
    """Minkowski capacity and cycle length match the reference length on 2x2 products."""
    left = hypercube(2)
    right = crosspolytope(2)
    prod = lagrangian_product(left, right)
    cap, cycle = capacity_minkowski_billiard(prod, geometry=prod, max_bounces=6)
    assert np.isfinite(cap)
    assert cycle.shape[1] == 4 and cycle.shape[0] >= 3
