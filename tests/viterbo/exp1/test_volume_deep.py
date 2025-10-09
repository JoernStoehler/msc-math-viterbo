from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.deep]

from viterbo.exp1.examples import regular_ngon2d
from viterbo.exp1.volume import volume


@pytest.mark.goal_math
def test_volume_triangulation_vs_fast_on_regular_octagon() -> None:
    """Triangulation and fast Delaunay-based estimators agree on an octagon."""
    P = regular_ngon2d(8)
    v_tria = float(volume(P, method="triangulation"))
    v_fast = float(volume(P, method="fast"))
    assert math.isclose(v_tria, v_fast, rel_tol=1e-12, abs_tol=1e-12)

