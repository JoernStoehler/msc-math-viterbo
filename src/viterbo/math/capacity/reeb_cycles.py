"""Reeb-cycle validation and oriented-edge graphs for modern polytopes (math layer)."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.math.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
)
from viterbo.math.geometry import enumerate_vertices
from viterbo.math.numerics import FACET_SOLVER_TOLERANCE, GEOMETRY_ABS_TOLERANCE


class OrientedEdge:
    """Directed edge on the Chaidez–Hutchings oriented-edge graph (compat shim)."""

    __slots__ = (
        "identifier",
        "facets",
        "tail_vertex",
        "head_vertex",
        "tail_missing_facet",
        "head_missing_facet",
    )

    def __init__(
        self,
        *,
        identifier: int,
        facets: tuple[int, int, int],
        tail_vertex: int,
        head_vertex: int,
        tail_missing_facet: int,
        head_missing_facet: int,
    ) -> None:
        """Create an oriented edge record with integer fields."""
        self.identifier = int(identifier)
        self.facets = facets
        self.tail_vertex = int(tail_vertex)
        self.head_vertex = int(head_vertex)
        self.tail_missing_facet = int(tail_missing_facet)
        self.head_missing_facet = int(head_missing_facet)


class OrientedEdgeGraph:
    """Oriented-edge transition graph together with adjacency metadata (compat shim)."""

    __slots__ = ("edges", "outgoing", "incoming", "dimension")

    def __init__(
        self,
        *,
        edges: tuple[OrientedEdge, ...],
        outgoing: tuple[tuple[int, ...], ...],
        incoming: tuple[tuple[int, ...], ...],
        dimension: int,
    ) -> None:
        """Construct a graph from edges and adjacency lists."""
        self.edges = edges
        self.outgoing = outgoing
        self.incoming = incoming
        self.dimension = int(dimension)

    def successors(self, edge_id: int) -> tuple[int, ...]:
        """Return successor edge identifiers for ``edge_id``."""
        return self.outgoing[int(edge_id)]

    def predecessors(self, edge_id: int) -> tuple[int, ...]:
        """Return predecessor edge identifiers for ``edge_id``."""
        return self.incoming[int(edge_id)]

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)


def build_oriented_edge_graph(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> OrientedEdgeGraph:
    """Construct the oriented-edge graph for ``bundle`` using combinatorics."""

    normals = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    dimension = int(normals.shape[1]) if normals.ndim == 2 else 0
    if dimension != 4:
        msg = "Combinatorial Reeb cycles are only implemented for dimension four."
        raise ValueError(msg)

    vertices = enumerate_vertices(normals, offsets, atol=atol)
    cone_vertices: list[jnp.ndarray] = []
    cone_active: list[tuple[int, ...]] = []
    for k in range(int(vertices.shape[0])):
        v = vertices[k]
        residuals = normals @ v - offsets
        active = jnp.where(jnp.abs(residuals) <= float(atol))[0]
        active_facets = tuple(int(i) for i in active.tolist())
        cone_vertices.append(v)
        cone_active.append(active_facets)
    vertex_lookup = {
        _vertex_key(cone_vertices[index], atol=atol): index for index in range(len(cone_vertices))
    }

    incident_edges: dict[int, list[int]] = {}
    reverse_incident: dict[int, list[int]] = {}
    triple_vertices: dict[tuple[int, int, int], list[int]] = {}
    missing_facets: dict[tuple[tuple[int, int, int], int], int] = {}

    for index in range(len(cone_vertices)):
        key = _vertex_key(cone_vertices[index], atol=atol)
        if key not in vertex_lookup:
            continue
        active = tuple(int(facet) for facet in cone_active[index])
        if len(active) != dimension:
            continue
        for triple in combinations(sorted(active), 3):
            remainder = sorted(set(active) - set(triple))
            if len(remainder) != 1:
                continue
            triple_vertices.setdefault(triple, []).append(index)
            missing_facets[(triple, index)] = remainder[0]

    edges_raw: list[tuple[int, int, int, int, int, int, int]] = []
    adjacency: dict[int, set[int]] = {}

    for triple, vertices in triple_vertices.items():
        if len(vertices) != 2:
            continue
        first, second = vertices
        tail_missing = missing_facets.get((triple, first))
        head_missing = missing_facets.get((triple, second))
        if tail_missing is None or head_missing is None:
            continue
        identifier = len(edges_raw)
        edges_raw.append(
            (first, second, triple[0], triple[1], triple[2], int(tail_missing), int(head_missing))
        )
        incident_edges.setdefault(first, []).append(identifier)
        reverse_incident.setdefault(second, []).append(identifier)

        identifier_rev = len(edges_raw)
        edges_raw.append(
            (second, first, triple[0], triple[1], triple[2], int(head_missing), int(tail_missing))
        )
        incident_edges.setdefault(second, []).append(identifier_rev)
        reverse_incident.setdefault(first, []).append(identifier_rev)

    for vertex_index in range(len(cone_vertices)):
        incoming = reverse_incident.get(vertex_index, [])
        outgoing = incident_edges.get(vertex_index, [])
        if not incoming or not outgoing:
            continue
        for source in incoming:
            edge_in = edges_raw[source]
            for target in outgoing:
                edge_out = edges_raw[target]
                if source == target:
                    continue
                if (edge_in[0] == edge_out[1]) and (edge_in[2:5] == edge_out[2:5]):
                    continue
                shared_facets = set(edge_in[2:5]).intersection(edge_out[2:5])
                if len(shared_facets) != 2:
                    continue
                if edge_in[6] == edge_out[5]:
                    continue
                adjacency.setdefault(source, set()).add(target)

    edge_count = len(edges_raw)
    outgoing = tuple(tuple(sorted(adjacency.get(index, set()))) for index in range(edge_count))
    incoming_map: dict[int, list[int]] = {index: [] for index in range(edge_count)}
    for source, targets in adjacency.items():
        for target in targets:
            incoming_map[target].append(source)
    incoming = tuple(tuple(sorted(incoming_map[index])) for index in range(edge_count))

    # Build compatibility objects lazily.
    edges: list[OrientedEdge] = []
    for idx, rec in enumerate(edges_raw):
        tail, head, f0, f1, f2, miss_tail, miss_head = rec
        edges.append(
            OrientedEdge(
                identifier=idx,
                facets=(int(f0), int(f1), int(f2)),
                tail_vertex=int(tail),
                head_vertex=int(head),
                tail_missing_facet=int(miss_tail),
                head_missing_facet=int(miss_head),
            )
        )
    return OrientedEdgeGraph(
        edges=tuple(edges), outgoing=outgoing, incoming=incoming, dimension=dimension
    )


def _vertex_key(vertex: Float[Array, " dimension"], *, atol: float) -> tuple[int, ...]:
    scaled = jnp.asarray(jnp.round(vertex / float(atol))).astype(int)
    return tuple(int(x) for x in scaled.tolist())


def _graph_diagnostics(graph: OrientedEdgeGraph) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    edge_count = graph.edge_count
    outgoing = graph.outgoing
    incoming = graph.incoming
    edges_without_succ = tuple(i for i, succ in enumerate(outgoing) if not succ)
    edges_without_pred = tuple(i for i, pred in enumerate(incoming) if not pred)
    return edge_count, edges_without_succ, edges_without_pred


def minimum_cycle_reference(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> Float[Array, " num_points dimension"]:
    """Return a representative closed orbit on the oriented-edge graph."""

    graph = build_oriented_edge_graph(normals, offsets, atol=atol)
    cycles = enumerate_simple_cycles(graph, limit=1)
    if not cycles:
        dimension = int(normals.shape[1]) if normals.ndim == 2 else 0
        return jnp.zeros((0, dimension), dtype=jnp.float64)

    cycle = cycles[0]
    vertices = enumerate_vertices(normals, offsets, atol=atol)
    points = [vertices[graph.edges[index].tail_vertex] for index in cycle[:-1]]
    return jnp.stack(points, axis=0)


def enumerate_simple_cycles(
    graph: OrientedEdgeGraph,
    *,
    limit: int,
) -> list[tuple[int, ...]]:
    """Enumerate simple cycles up to ``limit`` using DFS with canonicalisation."""

    results: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def canonicalize(sequence: Sequence[int]) -> tuple[int, ...]:
        if not sequence:
            return tuple()
        base = list(sequence[:-1])
        rotations = [tuple(base[i:] + base[:i]) for i in range(len(base))]
        canonical = min(rotations)
        return canonical + (canonical[0],)

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


def ehz_capacity_reference_reeb(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Reference capacity after validating the oriented-edge graph."""

    graph = build_oriented_edge_graph(normals, offsets, atol=atol)
    edge_count, edges_without_succ, edges_without_pred = _graph_diagnostics(graph)
    if edge_count == 0:
        raise ValueError(
            "Oriented-edge graph is empty; polytope lacks admissible edges. "
            f"edges_without_successors={edges_without_succ}, "
            f"edges_without_predecessors={edges_without_pred}"
        )
    return float(ehz_capacity_reference_facet_normals(normals, offsets, tol=tol))


def ehz_capacity_fast_reeb(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Fast capacity via facet solver with oriented-edge validation."""

    graph = build_oriented_edge_graph(normals, offsets, atol=atol)
    edge_count, edges_without_succ, edges_without_pred = _graph_diagnostics(graph)
    if edge_count == 0:
        raise ValueError(
            "Oriented-edge graph is empty; polytope lacks admissible edges. "
            f"edges_without_successors={edges_without_succ}, "
            f"edges_without_predecessors={edges_without_pred}"
        )
    try:
        return float(ehz_capacity_fast_facet_normals(normals, offsets, tol=tol))
    except ValueError:
        return float(ehz_capacity_reference_facet_normals(normals, offsets, tol=tol))
