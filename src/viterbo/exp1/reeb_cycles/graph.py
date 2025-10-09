from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.exp1.halfspaces import enumerate_vertices


@dataclass(frozen=True)
class OrientedEdge:
    identifier: int
    facets: tuple[int, int, int]
    tail_vertex: int
    head_vertex: int
    tail_missing_facet: int
    head_missing_facet: int


@dataclass(frozen=True)
class OrientedEdgeGraph:
    vertices: Float[Array, " k 4"]
    graph: dict[int, set[int]]
    edges: tuple[OrientedEdge, ...]
    dimension: int

    def outgoing(self, edge_id: int) -> list[int]:
        return sorted(self.graph.get(edge_id, set()))

    def incoming(self, edge_id: int) -> list[int]:
        return [src for src, dsts in self.graph.items() if edge_id in dsts]


def build_oriented_edge_graph(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    atol: float = 1e-9,
) -> OrientedEdgeGraph:
    A = jnp.asarray(normals, dtype=jnp.float64)
    b = jnp.asarray(offsets, dtype=jnp.float64)
    dim = int(A.shape[1])
    if dim != 4:
        raise ValueError("Oriented-edge graph is only implemented for dimension four.")

    all_verts = enumerate_vertices(A, b, atol=atol)
    # Build list of simple vertices and their active sets
    simple_vertices: list[Array] = []
    active_by_vertex: list[tuple[int, ...]] = []
    for vidx in range(int(all_verts.shape[0])):
        v = all_verts[vidx]
        residuals = A @ v - b
        act = tuple(int(i) for i in jnp.where(jnp.abs(residuals) <= float(atol))[0].tolist())
        if len(act) != dim:
            continue
        simple_vertices.append(v)
        active_by_vertex.append(act)

    triple_vertices: dict[tuple[int, int, int], list[int]] = {}
    missing_facet: dict[tuple[tuple[int, int, int], int], int] = {}
    for vidx, facets in enumerate(active_by_vertex):
        for triple in combinations(sorted(facets), 3):
            remainder = sorted(set(facets) - set(triple))
            if len(remainder) != 1:
                continue
            triple_vertices.setdefault(triple, []).append(vidx)
            missing_facet[(triple, vidx)] = remainder[0]

    edges: list[OrientedEdge] = []
    for triple, vids in triple_vertices.items():
        if len(vids) != 2:
            continue
        a, bidx = vids
        ta = missing_facet.get((triple, a))
        tb = missing_facet.get((triple, bidx))
        if ta is None or tb is None:
            continue
        eid = len(edges)
        edges.append(
            OrientedEdge(
                identifier=eid,
                facets=(int(triple[0]), int(triple[1]), int(triple[2])),
                tail_vertex=a,
                head_vertex=bidx,
                tail_missing_facet=ta,
                head_missing_facet=tb,
            )
        )
        eid2 = len(edges)
        edges.append(
            OrientedEdge(
                identifier=eid2,
                facets=(int(triple[0]), int(triple[1]), int(triple[2])),
                tail_vertex=bidx,
                head_vertex=a,
                tail_missing_facet=tb,
                head_missing_facet=ta,
            )
        )

    graph: dict[int, set[int]] = {}
    for e_in in edges:
        for e_out in edges:
            if e_in.identifier == e_out.identifier:
                continue
            if e_in.tail_vertex == e_out.head_vertex and e_in.facets == e_out.facets:
                continue
            shared = set(e_in.facets).intersection(e_out.facets)
            if len(shared) != 2:
                continue
            if e_in.head_missing_facet == e_out.tail_missing_facet:
                continue
            graph.setdefault(e_in.identifier, set()).add(e_out.identifier)

    if simple_vertices:
        V = jnp.stack(simple_vertices, axis=0)
    else:
        V = jnp.zeros((0, 4), dtype=jnp.float64)
    return OrientedEdgeGraph(vertices=V, graph=graph, edges=tuple(edges), dimension=dim)
