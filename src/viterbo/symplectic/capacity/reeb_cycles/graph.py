"""Oriented-edge graph construction for combinatorial Reeb cycles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import jax.numpy as jnp
import networkx as nx
from jaxtyping import Array, Float

from viterbo.geometry.polytopes import Polytope, polytope_combinatorics


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
    """Container bundling the oriented-edge graph with metadata."""

    graph: nx.DiGraph[int]
    edges: tuple[OrientedEdge, ...]
    dimension: int

    def outgoing(self, edge_id: int) -> list[int]:
        """Return identifiers of edges admissible after ``edge_id``."""
        return list(self.graph.successors(edge_id))

    def incoming(self, edge_id: int) -> list[int]:
        """Return identifiers of edges leading into ``edge_id``."""
        return list(self.graph.predecessors(edge_id))


def _vertex_key(vertex: jnp.ndarray, *, atol: float) -> tuple[int, ...]:
    scaled = jnp.asarray(jnp.round(vertex / float(atol))).astype(int)
    return tuple(int(x) for x in scaled.tolist())


def build_oriented_edge_graph(
    B_matrix: Float[Array, " num_facets dimension"],
    c_vector: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> OrientedEdgeGraph:
    """Construct the oriented-edge transition graph from ``(B, c)``."""
    B = jnp.asarray(B_matrix, dtype=jnp.float64)
    c = jnp.asarray(c_vector, dtype=jnp.float64)
    if B.ndim != 2:
        msg = "Facet matrix must be two-dimensional."
        raise ValueError(msg)
    if c.ndim != 1 or c.shape[0] != B.shape[0]:
        msg = "Offsets must match the number of facets."
        raise ValueError(msg)

    dimension = int(B.shape[1])
    if dimension != 4:
        msg = "Combinatorial Reeb cycles are only implemented for dimension four."
        raise ValueError(msg)

    polytope = Polytope(name="temporary-reeb", B=B, c=c)
    combinatorics = polytope_combinatorics(polytope, atol=atol, use_cache=False)

    vertex_lookup: dict[tuple[int, ...], int] = {}
    for index, vertex in enumerate(combinatorics.vertices):
        vertex_lookup[_vertex_key(vertex, atol=atol)] = index

    incident_edges: dict[int, list[int]] = defaultdict(list)
    reverse_incident: dict[int, list[int]] = defaultdict(list)
    triple_vertices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    missing_facets: dict[tuple[tuple[int, int, int], int], int] = {}

    for cone in combinatorics.normal_cones:
        key = _vertex_key(cone.vertex, atol=atol)
        if key not in vertex_lookup:
            continue
        vertex_index = vertex_lookup[key]
        active = tuple(cone.active_facets)
        if len(active) != dimension:
            # Chaidez–Hutchings assumes simple polytopes; skip degenerate vertices.
            continue
        for triple in combinations(sorted(active), 3):
            remainder = sorted(set(active) - set(triple))
            if len(remainder) != 1:
                continue
            triple_vertices[triple].append(vertex_index)
            missing_facets[(triple, vertex_index)] = remainder[0]

    oriented_edges: list[OrientedEdge] = []
    graph: nx.DiGraph[int] = nx.DiGraph()

    for triple, vertices in triple_vertices.items():
        if len(vertices) != 2:
            continue
        first, second = vertices
        tail_missing = missing_facets.get((triple, first))
        head_missing = missing_facets.get((triple, second))
        if tail_missing is None or head_missing is None:
            continue
        identifier = len(oriented_edges)
        edge = OrientedEdge(
            identifier=identifier,
            facets=triple,
            tail_vertex=first,
            head_vertex=second,
            tail_missing_facet=tail_missing,
            head_missing_facet=head_missing,
        )
        oriented_edges.append(edge)
        graph.add_node(identifier)
        incident_edges[first].append(identifier)
        reverse_incident[second].append(identifier)

        identifier_rev = len(oriented_edges)
        reverse_edge = OrientedEdge(
            identifier=identifier_rev,
            facets=triple,
            tail_vertex=second,
            head_vertex=first,
            tail_missing_facet=head_missing,
            head_missing_facet=tail_missing,
        )
        oriented_edges.append(reverse_edge)
        graph.add_node(identifier_rev)
        incident_edges[second].append(identifier_rev)
        reverse_incident[first].append(identifier_rev)

    for vertex_index in range(len(combinatorics.vertices)):
        incoming = reverse_incident.get(vertex_index, [])
        outgoing = incident_edges.get(vertex_index, [])
        if not incoming or not outgoing:
            continue
        for source in incoming:
            edge_in = oriented_edges[source]
            for target in outgoing:
                edge_out = oriented_edges[target]
                if source == target:
                    continue
                if (
                    edge_in.tail_vertex == edge_out.head_vertex
                    and edge_in.facets == edge_out.facets
                ):
                    continue
                shared_facets = set(edge_in.facets).intersection(edge_out.facets)
                if len(shared_facets) != 2:
                    continue
                if edge_in.head_missing_facet == edge_out.tail_missing_facet:
                    continue
                graph.add_edge(source, target)

    return OrientedEdgeGraph(
        graph=graph,
        edges=tuple(oriented_edges),
        dimension=dimension,
    )


__all__: Final = ["OrientedEdge", "OrientedEdgeGraph", "build_oriented_edge_graph"]
