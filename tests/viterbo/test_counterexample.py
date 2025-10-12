"""Haim–Kislev–Ostrover counterexample (4D) — systolic ratio > 1.

Build the official counterexample family K × T where K is a regular pentagon
in R^2 and T is K rotated by 90 degrees, then take the Lagrangian product in R^4.
Reference: docs/papers/2024-ostrover-counterexample-viterbo/main.tex:110–116.
"""

from __future__ import annotations

import math

import pytest

from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals
from viterbo.math.generators import counterexample_hk_ostrover_4d
from viterbo.math.volume import polytope_volume_reference


@pytest.mark.goal_math
@pytest.mark.deep
def test_counterexample_pentagon_product_sys_ratio_exceeds_one() -> None:
    """Official pentagon product K×T (90° rotation) violates Viterbo's bound."""
    verts4, B4, c4 = counterexample_hk_ostrover_4d(rotation=math.pi / 2.0)
    cap = ehz_capacity_reference_facet_normals(B4, c4)
    vol = polytope_volume_reference(B4, c4)
    sys = (cap * cap) / (2.0 * vol)
    assert sys > 1.0
