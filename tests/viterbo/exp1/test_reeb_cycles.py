from __future__ import annotations

import pytest

pytestmark = [pytest.mark.smoke]

from viterbo.exp1.examples import viterbo_counterexample
from viterbo.exp1.polytopes import to_halfspaces
from viterbo.exp1.reeb_cycles.graph import build_oriented_edge_graph


@pytest.mark.goal_math
def test_reeb_oriented_edge_graph_has_admissible_edges() -> None:
    """Oriented-edge graph for a 4D product has non-empty admissible edges (dim=4)."""
    prod = viterbo_counterexample()
    H = to_halfspaces(prod)
    A, b = H.as_tuple()
    og = build_oriented_edge_graph(A, b, atol=1e-9)
    assert og.dimension == 4
    assert len(og.edges) > 0
    assert og.vertices.shape[1] == 4
