from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.examples import hypercube
from viterbo.exp1.volume import volume


@pytest.mark.goal_math
def test_volume_matches_reference_on_hypercube_2d() -> None:
    """Volume for a 2D hypercube matches the known reference value."""
    expected = 4.0  # area of square with side length 2
    val = float(volume(hypercube(2), method="fast"))
    assert math.isclose(val, expected, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.goal_math
def test_volume_triangulation_on_hypercube_2d() -> None:
    """Triangulation method equals the known area for a 2D hypercube."""
    expected = 4.0
    val = float(volume(hypercube(2), method="triangulation"))
    assert math.isclose(val, expected, rel_tol=1e-10, abs_tol=1e-12)


@pytest.mark.goal_math
def test_volume_monte_carlo_on_hypercube_2d() -> None:
    """Monte Carlo method matches exact area on axis-aligned hypercube (box == hull)."""
    expected = 4.0
    val = float(volume(hypercube(2), method="monte_carlo"))
    assert math.isclose(val, expected, rel_tol=1e-9, abs_tol=1e-9)
