"""Reference enumeration of closed (K, T)-Minkowski billiard paths."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.polytopes import Polytope, polytope_combinatorics
from viterbo.symplectic.capacity.minkowski_billiards.fan import (
    MinkowskiNormalFan,
    build_normal_fan,
)
from viterbo.symplectic.core import support_function


def compute_minkowski_billiard_length_reference(
    billiard_table: Polytope,
    geometry: Polytope,
    *,
    max_bounces: int | None = None,
    atol: float = 1e-9,
) -> float:
    """Return the minimal closed path length for the Minkowski billiard."""
    fan = build_normal_fan(billiard_table, atol=atol)
    if fan.vertex_count == 0:
        raise ValueError("Normal fan construction yielded no vertices.")

    combinatorics_geometry = polytope_combinatorics(geometry, atol=atol)
    geometry_vertices = combinatorics_geometry.vertices

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
    """Enumerate simple cycles using a Johnson-style search with pruning."""
    neighbor_lists = [sorted(neighbors) for neighbors in fan.neighbors]
    vertex_count = fan.vertex_count
    seen: set[tuple[int, ...]] = set()

    def unblock(vertex: int, blocked: list[bool], block_map: list[set[int]]) -> None:
        if not blocked[vertex]:
            return
        blocked[vertex] = False
        while block_map[vertex]:
            neighbor = block_map[vertex].pop()
            if blocked[neighbor]:
                unblock(neighbor, blocked, block_map)

    stack: deque[int] = deque()

    def circuit(
        vertex: int,
        start: int,
        allowed: set[int],
        blocked: list[bool],
        block_map: list[set[int]],
    ) -> Iterable[tuple[int, ...]]:
        found = False
        stack.append(vertex)
        blocked[vertex] = True

        for neighbor in neighbor_lists[vertex]:
            if neighbor < start or neighbor not in allowed:
                continue

            if neighbor == start:
                if 3 <= len(stack) <= max_length:
                    cycle = tuple(stack)
                    canonical = _canonical_cycle(cycle)
                    if canonical not in seen:
                        seen.add(canonical)
                        yield cycle
                        found = True
                continue

            if len(stack) >= max_length or blocked[neighbor]:
                continue

            yielded = False
            for cycle in circuit(neighbor, start, allowed, blocked, block_map):
                yield cycle
                yielded = True
            if yielded:
                found = True

        if found:
            unblock(vertex, blocked, block_map)
        else:
            for neighbor in neighbor_lists[vertex]:
                if neighbor < start or neighbor not in allowed:
                    continue
                block_map[neighbor].add(vertex)

        stack.pop()
        return

    for start in range(vertex_count):
        if not neighbor_lists[start]:
            continue

        component = _component_containing(start, neighbor_lists, start)
        if component is None or len(component) < 3:
            for vertex in range(vertex_count):
                neighbor_lists[vertex] = [
                    neighbor for neighbor in neighbor_lists[vertex] if neighbor != start
                ]
            neighbor_lists[start] = []
            continue

        blocked = [False] * vertex_count
        block_map = [set() for _ in range(vertex_count)]

        for cycle in circuit(start, start, component, blocked, block_map):
            yield cycle

        for vertex in range(vertex_count):
            neighbor_lists[vertex] = [
                neighbor for neighbor in neighbor_lists[vertex] if neighbor != start
            ]
        neighbor_lists[start] = []


def _component_containing(
    start: int,
    neighbor_lists: list[list[int]],
    min_vertex: int,
) -> set[int] | None:
    """Return the SCC containing ``start`` restricted to vertices ≥ ``min_vertex``."""
    components = _strongly_connected_components(neighbor_lists, min_vertex)
    for component in components:
        if start in component:
            return component
    return None


def _strongly_connected_components(
    neighbor_lists: list[list[int]],
    min_vertex: int,
) -> list[set[int]]:
    """Compute SCCs of the induced subgraph on vertices ≥ ``min_vertex``."""
    vertex_count = len(neighbor_lists)
    index = 0
    stack: list[int] = []
    on_stack = [False] * vertex_count
    indices = [-1] * vertex_count
    lowlinks = [0] * vertex_count
    components: list[set[int]] = []

    def strongconnect(vertex: int) -> None:
        nonlocal index
        indices[vertex] = index
        lowlinks[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack[vertex] = True

        for neighbor in neighbor_lists[vertex]:
            if neighbor < min_vertex:
                continue
            if indices[neighbor] == -1:
                strongconnect(neighbor)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[neighbor])
            elif on_stack[neighbor]:
                lowlinks[vertex] = min(lowlinks[vertex], indices[neighbor])

        if lowlinks[vertex] == indices[vertex]:
            component: set[int] = set()
            while True:
                neighbor = stack.pop()
                on_stack[neighbor] = False
                component.add(neighbor)
                if neighbor == vertex:
                    break
            components.append(component)

    for vertex in range(min_vertex, vertex_count):
        if indices[vertex] != -1:
            continue
        if vertex < min_vertex:
            continue
        if not any(neighbor >= min_vertex for neighbor in neighbor_lists[vertex]):
            continue
        strongconnect(vertex)

    return components


def _canonical_cycle(sequence: tuple[int, ...]) -> tuple[int, ...]:
    length = len(sequence)
    if length == 0:
        return sequence

    candidates = []
    for shift in range(length):
        rotated = tuple(sequence[(shift + offset) % length] for offset in range(length))
        candidates.append(rotated)
        candidates.append(tuple(reversed(rotated)))
    return min(candidates)


def _cycle_length(
    cycle: tuple[int, ...],
    vertices: Float[Array, " num_vertices dimension"],
    geometry_vertices: Float[Array, " num_vertices_t dimension"],
) -> float:
    total = 0.0
    size = len(cycle)
    for index in range(size):
        start = cycle[index]
        end = cycle[(index + 1) % size]
        displacement = vertices[end] - vertices[start]
        segment_length = support_function(geometry_vertices, displacement)
        total += float(segment_length)
    return total
