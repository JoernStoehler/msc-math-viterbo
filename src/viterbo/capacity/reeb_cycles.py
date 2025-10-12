"""Reeb-cycle validation and oriented-edge graphs for modern polytopes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geom import Polytope as _GeometryPolytope
from viterbo.geom import polytope_combinatorics
from viterbo.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
)
from viterbo.types import Polytope
from viterbo.numerics import (
    FACET_SOLVER_TOLERANCE,
    GEOMETRY_ABS_TOLERANCE,
)


@dataclass(frozen=True)
class OrientedEdge:
    """Directed edge on the Chaidez–Hutchings oriented-edge graph."""

    identifier: int
    facets: tuple[int, int, int]
    tail_vertex: int
    head_vertex: int
    tail_missing_facet: int
    head_missing_facet: int


@dataclass(frozen=True)
class OrientedEdgeGraph:
    """Oriented-edge transition graph together with adjacency metadata."""

    edges: tuple[OrientedEdge, ...]
    outgoing: tuple[tuple[int, ...], ...]
    incoming: tuple[tuple[int, ...], ...]
    dimension: int

    def successors(self, edge_id: int) -> tuple[int, ...]:
        return self.outgoing[edge_id]

    def predecessors(self, edge_id: int) -> tuple[int, ...]:
        return self.incoming[edge_id]

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class OrientedEdgeDiagnostics:
    """Metadata describing adjacency defects in the oriented-edge graph."""

    edge_count: int
    edges_without_successors: tuple[int, ...]
    edges_without_predecessors: tuple[int, ...]


def _to_geometry_polytope(bundle: Polytope) -> _GeometryPolytope:
    normals = jnp.asarray(bundle.normals, dtype=jnp.float64)
    offsets = jnp.asarray(bundle.offsets, dtype=jnp.float64)
    return _GeometryPolytope(name="modern-reeb", B=normals, c=offsets)


def build_oriented_edge_graph(
    bundle: Polytope,
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> OrientedEdgeGraph:
    """Construct the oriented-edge graph for ``bundle`` using combinatorics."""

    geometry_polytope = _to_geometry_polytope(bundle)
    dimension = geometry_polytope.dimension
    if dimension != 4:
        msg = "Combinatorial Reeb cycles are only implemented for dimension four."
        raise ValueError(msg)

    combinatorics = polytope_combinatorics(geometry_polytope, atol=atol, use_cache=False)
    cones = combinatorics.normal_cones
    vertex_lookup = {
        _vertex_key(cone.vertex, atol=atol): index for index, cone in enumerate(cones)
    }

    incident_edges: dict[int, list[int]] = {}
    reverse_incident: dict[int, list[int]] = {}
    triple_vertices: dict[tuple[int, int, int], list[int]] = {}
    missing_facets: dict[tuple[tuple[int, int, int], int], int] = {}

    for index, cone in enumerate(cones):
        key = _vertex_key(cone.vertex, atol=atol)
        if key not in vertex_lookup:
            continue
        active = tuple(int(facet) for facet in cone.active_facets)
        if len(active) != dimension:
            continue
        for triple in combinations(sorted(active), 3):
            remainder = sorted(set(active) - set(triple))
            if len(remainder) != 1:
                continue
            triple_vertices.setdefault(triple, []).append(index)
            missing_facets[(triple, index)] = remainder[0]

    edges: list[OrientedEdge] = []
    adjacency: dict[int, set[int]] = {}

    for triple, vertices in triple_vertices.items():
        if len(vertices) != 2:
            continue
        first, second = vertices
        tail_missing = missing_facets.get((triple, first))
        head_missing = missing_facets.get((triple, second))
        if tail_missing is None or head_missing is None:
            continue
        identifier = len(edges)
        edges.append(
            OrientedEdge(
                identifier=identifier,
                facets=triple,
                tail_vertex=first,
                head_vertex=second,
                tail_missing_facet=tail_missing,
                head_missing_facet=head_missing,
            )
        )
        incident_edges.setdefault(first, []).append(identifier)
        reverse_incident.setdefault(second, []).append(identifier)

        identifier_rev = len(edges)
        edges.append(
            OrientedEdge(
                identifier=identifier_rev,
                facets=triple,
                tail_vertex=second,
                head_vertex=first,
                tail_missing_facet=head_missing,
                head_missing_facet=tail_missing,
            )
        )
        incident_edges.setdefault(second, []).append(identifier_rev)
        reverse_incident.setdefault(first, []).append(identifier_rev)

    for vertex_index in range(len(cones)):
        incoming = reverse_incident.get(vertex_index, [])
        outgoing = incident_edges.get(vertex_index, [])
        if not incoming or not outgoing:
            continue
        for source in incoming:
            edge_in = edges[source]
            for target in outgoing:
                edge_out = edges[target]
                if source == target:
                    continue
                if edge_in.tail_vertex == edge_out.head_vertex and edge_in.facets == edge_out.facets:
                    continue
                shared_facets = set(edge_in.facets).intersection(edge_out.facets)
                if len(shared_facets) != 2:
                    continue
                if edge_in.head_missing_facet == edge_out.tail_missing_facet:
                    continue
                adjacency.setdefault(source, set()).add(target)

    edge_count = len(edges)
    outgoing = tuple(
        tuple(sorted(adjacency.get(index, set()))) for index in range(edge_count)
    )
    incoming_map: dict[int, list[int]] = {index: [] for index in range(edge_count)}
    for source, targets in adjacency.items():
        for target in targets:
            incoming_map[target].append(source)
    incoming = tuple(tuple(sorted(incoming_map[index])) for index in range(edge_count))

    graph = OrientedEdgeGraph(
        edges=tuple(edges),
        outgoing=outgoing,
        incoming=incoming,
        dimension=dimension,
    )

    return graph


def _vertex_key(vertex: Float[Array, " dimension"], *, atol: float) -> tuple[int, ...]:
    scaled = jnp.asarray(jnp.round(vertex / float(atol))).astype(int)
    return tuple(int(x) for x in scaled.tolist())


def _graph_diagnostics(graph: OrientedEdgeGraph) -> OrientedEdgeDiagnostics:
    return OrientedEdgeDiagnostics(
        edge_count=graph.edge_count,
        edges_without_successors=tuple(
            edge.identifier for edge in graph.edges if not graph.successors(edge.identifier)
        ),
        edges_without_predecessors=tuple(
            index for index, predecessors in enumerate(graph.incoming) if not predecessors
        ),
    )


def ehz_capacity_reference_reeb(
    bundle: Polytope,
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Reference capacity after validating the oriented-edge graph."""

    graph = build_oriented_edge_graph(bundle, atol=atol)
    diagnostics = _graph_diagnostics(graph)
    if diagnostics.edge_count == 0:
        raise ValueError(
            "Oriented-edge graph is empty; polytope lacks admissible edges. "
            f"edges_without_successors={diagnostics.edges_without_successors}, "
            f"edges_without_predecessors={diagnostics.edges_without_predecessors}"
        )
    return float(ehz_capacity_reference_facet_normals(bundle, tol=tol))


def ehz_capacity_fast_reeb(
    bundle: Polytope,
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Fast capacity via facet solver with oriented-edge validation."""

    graph = build_oriented_edge_graph(bundle, atol=atol)
    diagnostics = _graph_diagnostics(graph)
    if diagnostics.edge_count == 0:
        raise ValueError(
            "Oriented-edge graph is empty; polytope lacks admissible edges. "
            f"edges_without_successors={diagnostics.edges_without_successors}, "
            f"edges_without_predecessors={diagnostics.edges_without_predecessors}"
        )
    try:
        return float(ehz_capacity_fast_facet_normals(bundle, tol=tol))
    except ValueError:
        return float(ehz_capacity_reference_facet_normals(bundle, tol=tol))


__all__ = [
    "OrientedEdge",
    "OrientedEdgeGraph",
    "OrientedEdgeDiagnostics",
    "build_oriented_edge_graph",
    "ehz_capacity_reference_reeb",
    "ehz_capacity_fast_reeb",
]
