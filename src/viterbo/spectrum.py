"""Action spectrum utilities built on the modern oriented-edge graph."""

from __future__ import annotations

from typing import Iterable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.math.capacity.reeb_cycles import (
    OrientedEdgeGraph,
    build_oriented_edge_graph,
)
from viterbo.math.numerics import GEOMETRY_ABS_TOLERANCE
from viterbo.math.geometry import enumerate_vertices


def ehz_spectrum_reference(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    head: int,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> Sequence[float]:
    """Return the leading EHZ actions by enumerating cycles on the oriented-edge graph."""

    graph = build_oriented_edge_graph(normals, offsets, atol=atol)
    vertices = enumerate_vertices(normals, offsets, atol=atol)
    cycles = _enumerate_simple_cycles(graph, limit=head)
    actions: list[float] = []
    for cycle in cycles:
        action = _cycle_action(cycle, graph, vertices)
        actions.append(action)
    return sorted(actions)[: int(head)]


def _enumerate_simple_cycles(graph: OrientedEdgeGraph, *, limit: int) -> Iterable[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    results: list[tuple[int, ...]] = []

    def canonicalize(sequence: Iterable[int]) -> tuple[int, ...]:
        seq = list(sequence)
        if not seq:
            return tuple()
        base = seq[:-1]
        rotations = [tuple(base[i:] + base[:i]) for i in range(len(base))]
        canon = min(rotations)
        return canon + (canon[0],)

    visiting: set[int] = set()

    def dfs(start: int, current: int, path: list[int]) -> None:
        if len(path) > graph.edge_count:
            return
        if len(results) >= limit:
            return
        visiting.add(current)
        for nxt in graph.successors(current):
            if nxt == start and len(path) >= 2:
                cycle = tuple(path + [start])
                key = canonicalize(cycle)
                if key not in seen:
                    seen.add(key)
                    results.append(cycle)
                    if len(results) >= limit:
                        visiting.discard(current)
                        return
                continue
            if nxt in visiting:
                continue
            dfs(start, nxt, path + [nxt])
        visiting.discard(current)

    for edge_id in range(graph.edge_count):
        if len(results) >= limit:
            break
        if not graph.successors(edge_id):
            continue
        dfs(edge_id, edge_id, [edge_id])

    return results


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


__all__ = [
    "ehz_spectrum_reference",
]
