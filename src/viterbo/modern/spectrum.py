"""Discrete action spectrum routines for the modern API.

First reference implementation targets 4D polytopes via the oriented-edge
graph used for Reeb cycles. Actions are computed as Euclidean cycle lengths
through polytope vertices (piecewise-linear boundary approximation). Other
dimensions are not yet implemented.
"""

from __future__ import annotations

from typing import Sequence, Iterable

from jaxtyping import Array, Float

from viterbo.modern.types import Polytope
import jax.numpy as jnp
from dataclasses import dataclass
import itertools


def ehz_spectrum_reference(bundle: Polytope, *, head: int) -> Sequence[float]:
    """Return the leading entries of the EHZ action spectrum (4D only).

    Returns up to ``head`` actions in ascending order. When fewer cycles are
    found, returns a shorter list.
    """
    d = int(bundle.vertices.shape[1])
    if d != 4:
        raise NotImplementedError("Spectrum reference is implemented for 4D only.")
    verts = jnp.asarray(bundle.vertices, dtype=jnp.float64)
    inc = jnp.asarray(bundle.incidence, dtype=bool)
    graph, edges = _build_oriented_edge_graph_from_incidence(verts, inc)
    cycles = _enumerate_simple_cycles(graph, max_length=12, limit=head)
    actions: list[float] = []
    for cyc in cycles:
        seq_vertices = [edges[eid].tail_vertex for eid in cyc[:-1]]  # drop closing duplicate
        if not seq_vertices:
            continue
        pts = verts[jnp.asarray(seq_vertices, dtype=jnp.int32)]
        x = pts
        x_next = jnp.concatenate([x[1:], x[:1]], axis=0)
        segs = jnp.linalg.norm(x_next - x, axis=1)
        actions.append(float(jnp.sum(segs)))
    actions_sorted = sorted(actions)
    return actions_sorted[: int(head)]


def ehz_spectrum_batched(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    head: int,
) -> Float[Array, " batch head"]:
    """Return padded EHZ spectra for each batch element.

    Padding semantics: use NaN for missing entries; no separate mask.
    For non-4D inputs or unbounded/infeasible systems, returns all-NaN rows.
    """
    batch = int(normals.shape[0])
    out = jnp.full((batch, head), float('nan'), dtype=jnp.float64)
    from viterbo.modern.polytopes import build_from_halfspaces

    for i in range(batch):
        B = jnp.asarray(normals[i], dtype=jnp.float64)
        c = jnp.asarray(offsets[i], dtype=jnp.float64)
        if B.shape[1] != 4:
            continue
        try:
            poly = build_from_halfspaces(B, c)
            seq = ehz_spectrum_reference(poly, head=head)
            if not seq:
                continue
            vals = jnp.asarray(seq, dtype=jnp.float64)
            if int(vals.shape[0]) < head:
                pad = jnp.full((head - int(vals.shape[0]),), float('nan'), dtype=jnp.float64)
                vals = jnp.concatenate([vals, pad], axis=0)
            out = out.at[i].set(vals[:head])
        except (ValueError, RuntimeError):
            continue
    return out


@dataclass(frozen=True)
class _Edge:
    identifier: int
    facets: tuple[int, int, int]
    tail_vertex: int
    head_vertex: int
    tail_missing_facet: int
    head_missing_facet: int


def _build_oriented_edge_graph_from_incidence(
    vertices: Float[Array, " v d"],
    incidence: Array,  # Bool[Array, " v m"]
) -> tuple[dict[int, set[int]], list[_Edge]]:
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    inc = jnp.asarray(incidence, dtype=bool)
    vcount = int(verts.shape[0])
    d = int(verts.shape[1])
    # Active facets per vertex
    active_sets: list[tuple[int, ...]] = []
    for vidx in range(vcount):
        act = tuple(int(i) for i in jnp.where(inc[vidx])[0].tolist())
        active_sets.append(act if len(act) == d else tuple())
    # Triples → vertices and missing facet mapping
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
    # Edges (both directions) and adjacency
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
    graph: dict[int, set[int]] = {}
    for vertex_index in range(vcount):
        inc_in = reverse_incident.get(vertex_index, [])
        inc_out = incident_edges.get(vertex_index, [])
        if not inc_in or not inc_out:
            continue
        for src in inc_in:
            e_in = edges[src]
            for dst in inc_out:
                e_out = edges[dst]
                if src == dst:
                    continue
                if e_in.tail_vertex == e_out.head_vertex and e_in.facets == e_out.facets:
                    continue
                if len(set(e_in.facets).intersection(e_out.facets)) != 2:
                    continue
                if e_in.head_missing_facet == e_out.tail_missing_facet:
                    continue
                graph.setdefault(src, set()).add(dst)
    return graph, edges


def _enumerate_simple_cycles(
    graph: dict[int, set[int]], *, max_length: int, limit: int
) -> list[list[int]]:
    """Enumerate simple directed cycles up to ``max_length`` using DFS.

    Returns cycles as edge-id lists, closed by repeating the first id at the end.
    Deduplicates cycles via canonical rotation of the edge sequence.
    """
    seen: set[tuple[int, ...]] = set()
    results: list[list[int]] = []

    def canonicalize(path: Iterable[int]) -> tuple[int, ...]:
        seq = list(path)
        if not seq:
            return tuple()
        base = seq[:-1]
        rotations = [tuple(base[i:] + base[:i]) for i in range(len(base))]
        canon = min(rotations)
        return canon + (canon[0],)

    visiting: set[int] = set()

    def dfs(start: int, current: int, path: list[int]) -> None:
        if len(path) > max_length:
            return
        if len(results) >= limit:
            return
        visiting.add(current)
        for nxt in sorted(graph.get(current, set())):
            if nxt == start and len(path) >= 2:
                cyc = path + [start]
                key = canonicalize(cyc)
                if key not in seen:
                    seen.add(key)
                    results.append(cyc)
                    if len(results) >= limit:
                        visiting.discard(current)
                        return
                continue
            if nxt in visiting:
                continue
            dfs(start, nxt, path + [nxt])
        visiting.discard(current)

    for node in list(graph.keys()):
        if len(results) >= limit:
            break
        dfs(node, node, [node])

    return results
