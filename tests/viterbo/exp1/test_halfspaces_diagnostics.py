from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.examples import hypercube
from viterbo.exp1.halfspaces import halfspace_degeneracy_metrics


@pytest.mark.goal_code
def test_halfspace_degeneracy_metrics_contract() -> None:
    """Degeneracy report returns sane metrics on a 2D hypercube (H-rep)."""
    H = hypercube(2)
    A, b = H.as_tuple()
    rep = halfspace_degeneracy_metrics(A, b, atol=1e-9)
    assert rep.m == 4 and rep.dim == 2 and rep.rank == 2
    assert math.isfinite(float(rep.min_singular_value)) and float(rep.min_singular_value) > 0.0
    assert math.isfinite(float(rep.condition_number)) and float(rep.condition_number) >= 1.0
    # Axis-aligned facets contain opposite-directed pairs → correlation ≤ 1
    assert float(rep.max_abs_row_correlation) <= 1.0
    assert math.isclose(float(rep.duplicate_facet_fraction), 0.0, rel_tol=0.0, abs_tol=0.0)


@pytest.mark.goal_code
def test_halfspace_degeneracy_metrics_with_vertices() -> None:
    """Vertex-level metrics are populated when requested (counts and conditioning)."""
    H = hypercube(2)
    A, b = H.as_tuple()
    rep = halfspace_degeneracy_metrics(A, b, with_vertices=True, atol=1e-9)
    assert rep.vertex_count is not None and rep.vertex_count >= 3
    assert rep.simple_vertex_count is not None and rep.simple_vertex_count >= 3
    assert rep.min_simple_vertex_sigma is not None and float(rep.min_simple_vertex_sigma) > 0.0
    assert rep.max_simple_vertex_condition is not None and float(rep.max_simple_vertex_condition) >= 1.0
