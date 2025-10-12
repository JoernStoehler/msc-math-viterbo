"""Minkowski billiard solvers for modern polytopes (math layer)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Iterator, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.math.numerics import GEOMETRY_ABS_TOLERANCE
from viterbo.math.geometry import enumerate_vertices


@dataclass(frozen=True)
class NormalCone:
    """Vertex together with the indices of its active facets."""

    vertex: Float[Array, " dimension"]
    active_facets: tuple[int, ...]


@dataclass(frozen=True)
class MinkowskiNormalFan:
    """Normal fan representation of a polytope."""

    normals: Float[Array, " num_facets dimension"]
    vertices: Float[Array, " num_vertices dimension"]
    cones: tuple[NormalCone, ...]
    adjacency: Array
    neighbors: tuple[tuple[int, ...], ...]
    coordinate_blocks: tuple[tuple[int, ...], ...]

    @property
    def dimension(self) -> int:
        """Ambient dimension of the normal fan vertices."""
        return int(self.vertices.shape[1])

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the normal fan."""
        return int(self.vertices.shape[0])


def build_normal_fan(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> MinkowskiNormalFan:
    """Construct the normal fan from a half-space description (B, c)."""
    normals = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    vertices = enumerate_vertices(normals, offsets, atol=atol)
    cones: list[NormalCone] = []
    for k in range(int(vertices.shape[0])):
        v = vertices[k]
        residuals = normals @ v - offsets
        active = jnp.where(jnp.abs(residuals) <= float(atol))[0]
        cones.append(
            NormalCone(vertex=v, active_facets=tuple(int(i) for i in active.tolist()))
        )
    if not cones:
        raise ValueError("Polytope must have at least one vertex to build a normal fan.")
    vertices = jnp.stack([cone.vertex for cone in cones], axis=0)
    adjacency = _vertex_adjacency(cones, dimension=int(vertices.shape[1]))
    neighbors = tuple(
        tuple(int(index) for index in jnp.where(row)[0].tolist()) for row in adjacency
    )
    coordinate_blocks = _coordinate_blocks(normals, tol=1e-12)
    return MinkowskiNormalFan(
        normals=normals,
        vertices=vertices,
        cones=tuple(cones),
        adjacency=adjacency,
        neighbors=neighbors,
        coordinate_blocks=coordinate_blocks,
    )


def _vertex_adjacency(cones: Sequence[NormalCone], *, dimension: int) -> Array:
    count = len(cones)
    adjacency = jnp.zeros((count, count), dtype=bool)
    for first_index, first_cone in enumerate(cones):
        facets_first = set(first_cone.active_facets)
        for second_index in range(first_index + 1, count):
            second_cone = cones[second_index]
            facets_second = set(second_cone.active_facets)
            shared = len(facets_first.intersection(facets_second))
            threshold = max(0, dimension - 1)
            if shared >= threshold:
                adjacency = adjacency.at[first_index, second_index].set(True)
                adjacency = adjacency.at[second_index, first_index].set(True)
    adjacency = adjacency.at[jnp.diag_indices(count)].set(False)
    return adjacency


def _coordinate_blocks(
    matrix: Float[Array, " num_facets dimension"], *, tol: float
) -> tuple[tuple[int, ...], ...]:
    dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    adjacency = jnp.zeros((dimension, dimension), dtype=bool)
    for row in matrix:
        support = jnp.where(jnp.abs(row) > tol)[0]
        if support.size <= 1:
            continue
        for first, second in combinations(support.tolist(), 2):
            adjacency = adjacency.at[first, second].set(True)
            adjacency = adjacency.at[second, first].set(True)
    visited = [False] * dimension
    blocks: list[tuple[int, ...]] = []
    for index in range(dimension):
        if visited[index]:
            continue
        stack = [index]
        component: list[int] = []
        visited[index] = True
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = jnp.where(adjacency[current])[0]
            for neighbor in neighbors.tolist():
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        blocks.append(tuple(sorted(component)))
    return tuple(blocks)


def _support_function(
    vertices: Float[Array, " num_vertices dimension"], direction: Float[Array, " dimension"]
) -> float:
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)
    if verts.shape[0] == 0:
        raise ValueError("Support function requires at least one vertex.")
    return float(jnp.max(verts @ direction))


def minkowski_billiard_length_reference(
    table_normals: Float[Array, " num_facets dimension"],
    table_offsets: Float[Array, " num_facets"],
    geometry_normals: Float[Array, " num_facets dimension"],
    geometry_offsets: Float[Array, " num_facets"],
    *,
    max_bounces: int | None = None,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> float:
    """Reference Minkowski billiard cycle length search by enumeration."""
    fan = build_normal_fan(table_normals, table_offsets, atol=atol)
    if fan.vertex_count == 0:
        raise ValueError("Normal fan construction yielded no vertices.")
    geometry_vertices = enumerate_vertices(geometry_normals, geometry_offsets, atol=atol)

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


def minkowski_billiard_length_fast(
    table_normals: Float[Array, " num_facets dimension"],
    table_offsets: Float[Array, " num_facets"],
    geometry_normals: Float[Array, " num_facets dimension"],
    geometry_offsets: Float[Array, " num_facets"],
    *,
    max_bounces: int | None = None,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> float:
    """Fast Minkowski billiard cycle length via pruned DFS and heuristics."""
    fan = build_normal_fan(table_normals, table_offsets, atol=atol)
    if fan.vertex_count == 0:
        raise ValueError("Normal fan construction yielded no vertices.")
    geometry_vertices = enumerate_vertices(geometry_normals, geometry_offsets, atol=atol)
    max_length = max_bounces if max_bounces is not None else (fan.dimension + 2)
    if max_length < 3:
        raise ValueError("Closed billiard paths require at least three bounces.")

    length_matrix, min_edge_length = _pairwise_lengths(fan, geometry_vertices)
    if not math.isfinite(min_edge_length):
        raise ValueError("No admissible edges available for Minkowski billiard paths.")

    best_overall = math.inf
    num_vertices = fan.vertex_count
    neighbors = fan.neighbors
    prefix_best: dict[tuple[int, int, int, int], float] = {}

    for start in range(num_vertices):
        if not neighbors[start]:
            continue
        start_mask = 1 << start
        best_holder = [best_overall]
        completion_bounds = _completion_bounds(
            start, neighbors, length_matrix, max_length=max_length
        )
        for total_vertices in range(3, max_length + 1):
            steps_remaining = total_vertices - 1
            _dfs_prefix(
                start=start,
                current=start,
                visited_mask=start_mask,
                steps_remaining=steps_remaining,
                current_length=0.0,
                neighbors=neighbors,
                length_matrix=length_matrix,
                min_edge_length=min_edge_length,
                completion_bounds=completion_bounds,
                best_overall_ref=best_holder,
                prefix_best=prefix_best,
            )
            best_overall = min(best_overall, best_holder[0])
            best_holder[0] = best_overall

    if not math.isfinite(best_overall):
        raise ValueError("No closed Minkowski billiard cycle satisfies the constraints.")

    return float(best_overall)


def _enumerate_cycles(
    fan: MinkowskiNormalFan,
    *,
    max_length: int,
) -> Iterable[tuple[int, ...]]:
    neighbor_lists: list[list[int]] = [sorted(neighbors) for neighbors in fan.neighbors]
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
    ) -> Iterator[tuple[int, ...]]:
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
        block_map: list[set[int]] = [set() for _ in range(vertex_count)]
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
    components = _strongly_connected_components(neighbor_lists, min_vertex)
    for component in components:
        if start in component:
            return component
    return None


def _strongly_connected_components(
    neighbor_lists: list[list[int]],
    min_vertex: int,
) -> list[set[int]]:
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
        if indices[vertex] == -1:
            strongconnect(vertex)
    return components


def _canonical_cycle(cycle: tuple[int, ...]) -> tuple[int, ...]:
    rotated = list(cycle)
    min_index = rotated.index(min(rotated))
    base = rotated[min_index:] + rotated[:min_index]
    return tuple(base)


def _cycle_length(
    cycle: tuple[int, ...],
    fan_vertices: Float[Array, " num_vertices dimension"],
    geometry_vertices: Float[Array, " num_vertices dimension"],
) -> float:
    total = 0.0
    length = len(cycle)
    if length < 3:
        return 0.0
    for i in range(length):
        a = fan_vertices[cycle[i]]
        b = fan_vertices[cycle[(i + 1) % length]]
        v = a - b
        length_segment = _support_function(geometry_vertices, v)
        total += float(length_segment)
    return total


def _pairwise_lengths(
    fan: MinkowskiNormalFan,
    geometry_vertices: Float[Array, " num_vertices dimension"],
) -> tuple[Array, float]:
    V = jnp.asarray(fan.vertices, dtype=jnp.float64)
    G = jnp.asarray(geometry_vertices, dtype=jnp.float64)
    n = V.shape[0]
    lengths = jnp.zeros((n, n), dtype=jnp.float64)
    min_edge = jnp.inf
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = V[i] - V[j]
            s = _support_function(G, d)
            lengths = lengths.at[i, j].set(s)
            min_edge = jnp.minimum(min_edge, s)
    return lengths, float(min_edge)


def _completion_bounds(
    start: int,
    neighbors: tuple[tuple[int, ...], ...],
    length_matrix: Array,
    *,
    max_length: int,
) -> list[float]:
    n = len(neighbors)
    bounds = [float("inf")] * n
    for i in range(n):
        if i == start:
            continue
        nearest = (
            float(jnp.min(length_matrix[i, jnp.array(neighbors[i])]))
            if neighbors[i]
            else float("inf")
        )
        bounds[i] = nearest
    return bounds


def _dfs_prefix(
    *,
    start: int,
    current: int,
    visited_mask: int,
    steps_remaining: int,
    current_length: float,
    neighbors: tuple[tuple[int, ...], ...],
    length_matrix: Array,
    min_edge_length: float,
    completion_bounds: list[float],
    best_overall_ref: list[float],
    prefix_best: dict[tuple[int, int, int, int], float],
) -> None:
    if steps_remaining == 0:
        if current in neighbors[start]:
            best_overall_ref[0] = min(
                best_overall_ref[0], current_length + float(length_matrix[current, start])
            )
        return
    bound = current_length + steps_remaining * min_edge_length
    if bound >= best_overall_ref[0]:
        return
    for nxt in neighbors[current]:
        if visited_mask & (1 << nxt):
            continue
        remaining = steps_remaining - 1
        new_length = current_length + float(length_matrix[current, nxt])
        key = (current, nxt, remaining, visited_mask | (1 << nxt))
        if key in prefix_best and prefix_best[key] <= new_length:
            continue
        if new_length + remaining * min_edge_length >= best_overall_ref[0]:
            continue
        prefix_best[key] = new_length
        _dfs_prefix(
            start=start,
            current=nxt,
            visited_mask=visited_mask | (1 << nxt),
            steps_remaining=remaining,
            current_length=new_length,
            neighbors=neighbors,
            length_matrix=length_matrix,
            min_edge_length=min_edge_length,
            completion_bounds=completion_bounds,
            best_overall_ref=best_overall_ref,
            prefix_best=prefix_best,
        )
