from __future__ import annotations

from collections import deque
from typing import Iterable

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.exp1.minkowski_billiards.fan import MinkowskiNormalFan


def compute_minkowski_billiard_length_reference(
    fan: MinkowskiNormalFan,
    geometry_vertices: Float[Array, " k dim"],
    *,
    max_bounces: int | None = None,
) -> float:
    """Return minimal closed cycle length for the Minkowski billiard on ``fan``.

    The geometry is provided as a vertex cloud for the support function.
    """
    dimension = fan.dimension
    max_length = max_bounces if max_bounces is not None else (dimension + 2)
    if max_length < 3:
        raise ValueError("Closed billiard paths require at least three bounces.")
    best = jnp.inf
    for cycle in _enumerate_cycles(fan, max_length=max_length):
        length = _cycle_length(cycle, fan.vertices, geometry_vertices)
        if length < best:
            best = length
    if not bool(jnp.isfinite(best)):
        raise ValueError("No closed Minkowski billiard cycle satisfies the constraints.")
    return float(best)


def _enumerate_cycles(
    fan: MinkowskiNormalFan,
    *,
    max_length: int,
) -> Iterable[tuple[int, ...]]:
    neighbor_lists: list[list[int]] = [sorted(nei) for nei in fan.neighbors]
    vcount = fan.vertex_count
    seen: set[tuple[int, ...]] = set()

    def unblock(v: int, blocked: list[bool], block_map: list[set[int]]) -> None:
        if not blocked[v]:
            return
        blocked[v] = False
        while block_map[v]:
            w = block_map[v].pop()
            if blocked[w]:
                unblock(w, blocked, block_map)

    stack: deque[int] = deque()

    def circuit(
        v: int,
        start: int,
        allowed: set[int],
        blocked: list[bool],
        block_map: list[set[int]],
    ) -> Iterable[tuple[int, ...]]:
        found = False
        stack.append(v)
        blocked[v] = True
        for w in neighbor_lists[v]:
            if w < start or w not in allowed:
                continue
            if w == start:
                if 3 <= len(stack) <= max_length:
                    cyc: tuple[int, ...] = tuple(stack)
                    canon = _canonical_cycle(cyc)
                    if canon not in seen:
                        seen.add(canon)
                        yield cyc
                        found = True
                continue
            if len(stack) >= max_length or blocked[w]:
                continue
            yielded = False
            for cyc in circuit(w, start, allowed, blocked, block_map):
                yield cyc
                yielded = True
            if yielded:
                found = True
        if found:
            unblock(v, blocked, block_map)
        else:
            for w in neighbor_lists[v]:
                if w < start or w not in allowed:
                    continue
                block_map[w].add(v)
        stack.pop()
        return

    for start in range(vcount):
        if not neighbor_lists[start]:
            continue
        component = _component_containing(start, neighbor_lists, start)
        if component is None or len(component) < 3:
            for v in range(vcount):
                neighbor_lists[v] = [n for n in neighbor_lists[v] if n != start]
            neighbor_lists[start] = []
            continue
        blocked = [False] * vcount
        block_map: list[set[int]] = [set() for _ in range(vcount)]
        for cyc in circuit(start, start, component, blocked, block_map):
            yield cyc
        for v in range(vcount):
            neighbor_lists[v] = [n for n in neighbor_lists[v] if n != start]
        neighbor_lists[start] = []


def _component_containing(
    start: int, neighbor_lists: list[list[int]], min_vertex: int
) -> set[int] | None:
    for comp in _strongly_connected_components(neighbor_lists, min_vertex):
        if start in comp:
            return comp
    return None


def _strongly_connected_components(
    neighbor_lists: list[list[int]], min_vertex: int
) -> list[set[int]]:
    n = len(neighbor_lists)
    index = 0
    stack: list[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlinks = [0] * n
    comps: list[set[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True
        for w in neighbor_lists[v]:
            if w < min_vertex:
                continue
            if indices[w] == -1:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            comp: set[int] = set()
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.add(w)
                if w == v:
                    break
            comps.append(comp)

    for v in range(min_vertex, n):
        if indices[v] != -1:
            continue
        if v < min_vertex:
            continue
        if not any(w >= min_vertex for w in neighbor_lists[v]):
            continue
        strongconnect(v)
    return comps


def _canonical_cycle(seq: tuple[int, ...]) -> tuple[int, ...]:
    if not seq:
        return seq
    candidates: list[tuple[int, ...]] = []
    L = len(seq)
    for s in range(L):
        rot = tuple(seq[(s + k) % L] for k in range(L))
        candidates.append(rot)
        candidates.append(tuple(reversed(rot)))
    return min(candidates)


def _cycle_length(
    cycle: tuple[int, ...],
    vertices: Float[Array, " k dim"],
    geometry_vertices: Float[Array, " k2 dim"],
) -> float:
    total = 0.0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        disp = vertices[b] - vertices[a]
        total += float(_support_function(geometry_vertices, disp))
    return total


def _support_function(
    V: Float[Array, " k dim"],
    d: Float[Array, " dim"],
) -> float:
    return float(jnp.max(V @ jnp.asarray(d, dtype=jnp.float64)))


# Public aliases to avoid private import in callers under strict typing.
def enumerate_cycles(fan: MinkowskiNormalFan, *, max_length: int) -> Iterable[tuple[int, ...]]:
    return _enumerate_cycles(fan, max_length=max_length)


def cycle_length(
    cycle: tuple[int, ...],
    vertices: Float[Array, " k dim"],
    geometry_vertices: Float[Array, " k2 dim"],
) -> float:
    return _cycle_length(cycle, vertices, geometry_vertices)
