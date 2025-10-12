"""EHZ spectrum utilities built on the oriented-edge graph."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.math.capacity.reeb_cycles import (
    OrientedEdgeGraph,
    build_oriented_edge_graph,
    enumerate_simple_cycles,
)
from viterbo.math.geometry import enumerate_vertices
from viterbo.math.numerics import GEOMETRY_ABS_TOLERANCE


def ehz_spectrum_reference(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    head: int,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> Iterable[float]:
    """Return the leading EHZ actions by enumerating oriented-edge cycles."""

    graph = build_oriented_edge_graph(normals, offsets, atol=atol)
    vertices = enumerate_vertices(normals, offsets, atol=atol)
    cycles = enumerate_simple_cycles(graph, limit=int(head))
    actions: list[float] = []
    for cycle in cycles:
        action = _cycle_action(cycle, graph, vertices)
        actions.append(action)
    return sorted(actions)[: int(head)]


def _cycle_action(
    cycle: tuple[int, ...],
    graph: OrientedEdgeGraph,
    vertices: Float[Array, " num_vertices dimension"],
) -> float:
    edges = graph.edges
    total = 0.0
    length = len(cycle)
    if length < 2:
        return 0.0
    for index in range(length - 1):
        current_edge = edges[cycle[index]]
        next_edge = edges[cycle[index + 1]]
        tail = int(current_edge.tail_vertex)
        head = int(next_edge.tail_vertex)
        point_tail = vertices[tail]
        point_head = vertices[head]
        total += float(jnp.linalg.norm(point_head - point_tail))
    return total
