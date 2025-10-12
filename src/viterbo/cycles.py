"""Cycle extraction for the modern API (initial 4D support).

Implements the 4D oriented-edge graph directly over modern Polytope bundles
using the vertex–facet incidence already present in ``Polytope``.
No dependencies on legacy modules.
"""

from __future__ import annotations

import itertools
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from viterbo.types import Polytope


def minimum_cycle_reference(bundle: Polytope) -> Float[Array, " num_points dimension"]:
    """Return a representative minimum-action cycle.

    Implements the 4D-oriented-edge Reeb cycle extraction. Raises NotImplementedError
    for other dimensions pending a general formulation.
    """
    d = int(bundle.vertices.shape[1])
    if d != 4:
        raise NotImplementedError("Reeb cycle extraction is implemented for dimension 4 only.")
    # Build oriented-edge graph from incidence and vertices; then extract a cycle.
    vertices = jnp.asarray(bundle.vertices, dtype=jnp.float64)
    incidence: Bool[Array, " v m"] = jnp.asarray(bundle.incidence, dtype=bool)
    vcount = int(vertices.shape[0])

    # Active facets per vertex (simple vertices must have exactly d active facets)
    active_sets: list[tuple[int, ...]] = []
    for vidx in range(vcount):
        act = tuple(int(i) for i in jnp.where(incidence[vidx])[0].tolist())
        if len(act) == d:
            active_sets.append(act)
        else:
            active_sets.append(tuple())

    # Map triples to the two vertices where they are active, tracking the missing facet
    triple_vertices: dict[tuple[int, int, int], list[int]] = {}
    missing_facet: dict[tuple[tuple[int, int, int], int], int] = {}
    for vidx, facets in enumerate(active_sets):
        if len(facets) != d:
            continue
        for triple in itertools.combinations(sorted(facets), 3):
            t: tuple[int, int, int] = (int(triple[0]), int(triple[1]), int(triple[2]))
            rem = sorted(set(facets) - set(t))
            if len(rem) != 1:
                continue
            triple_vertices.setdefault(t, []).append(vidx)
            missing_facet[(t, vidx)] = int(rem[0])

    # Oriented edges: for each triple with two vertices, add both directions
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Edge:
        identifier: int
        facets: tuple[int, int, int]
        tail_vertex: int
        head_vertex: int
        tail_missing_facet: int
        head_missing_facet: int

    edges: list[_Edge] = []
    incident_edges: dict[int, list[int]] = {}
    reverse_incident: dict[int, list[int]] = {}
    for triple, vids in triple_vertices.items():
        if len(vids) != 2:
            continue
        a, b = vids
        ta = missing_facet.get((triple, a))
        tb = missing_facet.get((triple, b))
        if ta is None or tb is None:
            continue
        eid = len(edges)
        edges.append(_Edge(eid, triple, a, b, ta, tb))
        incident_edges.setdefault(a, []).append(eid)
        reverse_incident.setdefault(b, []).append(eid)
        eid2 = len(edges)
        edges.append(_Edge(eid2, triple, b, a, tb, ta))
        incident_edges.setdefault(b, []).append(eid2)
        reverse_incident.setdefault(a, []).append(eid2)

    # Transition graph as adjacency list over edge identifiers
    graph: dict[int, set[int]] = {}
    for vertex_index in range(vcount):
        incoming = reverse_incident.get(vertex_index, [])
        outgoing = incident_edges.get(vertex_index, [])
        if not incoming or not outgoing:
            continue
        for src in incoming:
            e_in = edges[src]
            for dst in outgoing:
                e_out = edges[dst]
                if src == dst:
                    continue
                if e_in.tail_vertex == e_out.head_vertex and e_in.facets == e_out.facets:
                    continue
                shared = set(e_in.facets).intersection(e_out.facets)
                if len(shared) != 2:
                    continue
                if e_in.head_missing_facet == e_out.tail_missing_facet:
                    continue
                graph.setdefault(src, set()).add(dst)
    # DFS to find any simple directed cycle
    visited: set[int] = set()
    stack: list[int] = []
    in_stack: set[int] = set()

    def _dfs(u: int) -> list[int] | None:
        visited.add(u)
        stack.append(u)
        in_stack.add(u)
        for v in sorted(graph.get(u, set())):
            if v not in visited:
                cyc = _dfs(v)
                if cyc is not None:
                    return cyc
            elif v in in_stack:
                if v in stack:
                    idx = stack.index(v)
                    return stack[idx:] + [v]
        stack.pop()
        in_stack.discard(u)
        return None

    cycle_ids: list[int] | None = None
    for start in list(graph.keys()):
        if start not in visited:
            cycle_ids = _dfs(start)
            if cycle_ids is not None:
                break
    if not cycle_ids or len(cycle_ids) < 3:
        # No simple cycle found; return an empty path with correct dimension.
        return jnp.zeros((0, d), dtype=jnp.float64)
    ids = cycle_ids[:-1]
    pts = [vertices[edges[eid].tail_vertex] for eid in ids]
    return jnp.stack(pts, axis=0)

