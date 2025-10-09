from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.examples import hypercube
from viterbo.exp1.polytopes import lagrangian_product, scale, to_vertices
from viterbo.exp1.systolic import systolic_ratio


@pytest.mark.goal_math
def test_systolic_ratio_invariant_under_scaling_for_product() -> None:
    """Systolic ratio stays invariant under uniform scaling of a 2x2 product."""
    left = to_vertices(hypercube(2))
    right = to_vertices(hypercube(2))
    prod = lagrangian_product(left, right)
    base = float(systolic_ratio(prod, method="auto"))
    scaled = scale(1.5, prod)
    val = float(systolic_ratio(scaled, method="auto"))
    assert math.isfinite(base) and math.isfinite(val)
    assert math.isclose(base, val, rel_tol=1e-9, abs_tol=1e-12)
    assert base > 0.0

