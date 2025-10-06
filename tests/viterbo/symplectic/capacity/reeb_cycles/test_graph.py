"""Graph construction checks for oriented-edge combinatorics."""

from __future__ import annotations

import pytest

from viterbo.geometry.polytopes import regular_polygon_product
from viterbo.symplectic.capacity.reeb_cycles.graph import build_oriented_edge_graph


def _assert_bidirectional(graph) -> None:
    for edge_id, edge in enumerate(graph.edges):
        successors = set(graph.outgoing(edge_id))
        predecessors = set(graph.incoming(edge_id))
        assert successors or predecessors


@pytest.mark.parametrize(
    "polytope_factory",
    [
        lambda: regular_polygon_product(5, 5, rotation=0.0, name="pentagon-product"),
        lambda: regular_polygon_product(6, 6, rotation=0.0, name="hexagon-product"),
    ],
)
def test_graph_has_edges(polytope_factory) -> None:
    polytope = polytope_factory()
    B, c = polytope.halfspace_data()
    graph = build_oriented_edge_graph(B, c)
    assert graph.graph.number_of_nodes() > 0
    assert graph.graph.number_of_edges() > 0
    _assert_bidirectional(graph)


def test_graph_requires_four_dimensions() -> None:
    from viterbo.geometry.polytopes import simplex_with_uniform_weights

    polytope = simplex_with_uniform_weights(2, name="triangle")
    B, c = polytope.halfspace_data()
    with pytest.raises(ValueError):
        build_oriented_edge_graph(B, c)
