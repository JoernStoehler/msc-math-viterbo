"""Minkowski billiard solvers for modern polytopes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Iterator, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.polytopes import (
    Polytope as _GeometryPolytope,
    polytope_combinatorics,
)
from viterbo.modern.numerics import GEOMETRY_ABS_TOLERANCE
from viterbo.modern.types import Polytope


@dataclass(frozen=True)
class MinkowskiNormalFan:
    """Normal fan representation of a polytope."""

    polytope: _GeometryPolytope
    vertices: Float[Array, " num_vertices dimension"]
    cones: tuple
    adjacency: Array
    neighbors: tuple[tuple[int, ...], ...]
    coordinate_blocks: tuple[tuple[int, ...], ...]

    @property
    def dimension(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def _to_geometry_polytope(bundle: Polytope, *, name: str) -> _GeometryPolytope:
    normals = jnp.asarray(bundle.normals, dtype=jnp.float64)
    offsets = jnp.asarray(bundle.offsets, dtype=jnp.float64)
    return _GeometryPolytope(name=name, B=normals, c=offsets)


def build_normal_fan(
    bundle: Polytope,
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> MinkowskiNormalFan:
    geometry_poly = _to_geometry_polytope(bundle, name="modern-billiard")
    combinatorics = polytope_combinatorics(geometry_poly, atol=atol)
    cones = combinatorics.normal_cones
    if not cones:
        raise ValueError("Polytope must have at least one vertex to build a normal fan.")
    vertices = jnp.stack([cone.vertex for cone in cones], axis=0)
    adjacency = _vertex_adjacency(cones, dimension=geometry_poly.dimension)
    neighbors = tuple(
        tuple(int(index) for index in jnp.where(row)[0].tolist()) for row in adjacency
    )
    coordinate_blocks = _coordinate_blocks(geometry_poly.B, tol=1e-12)
    return MinkowskiNormalFan(
        polytope=geometry_poly,
        vertices=vertices,
        cones=tuple(cones),
        adjacency=adjacency,
        neighbors=neighbors,
        coordinate_blocks=coordinate_blocks,
    )


def _vertex_adjacency(cones: Sequence, *, dimension: int) -> Array:
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


def _coordinate_blocks(matrix: Float[Array, " num_facets dimension"], *, tol: float) -> tuple[tuple[int, ...], ...]:
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


def _support_function(vertices: Float[Array, " num_vertices dimension"], direction: Float[Array, " dimension"]) -> float:
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)
    if verts.shape[0] == 0:
        raise ValueError("Support function requires at least one vertex.")
    return float(jnp.max(verts @ direction))


def minkowski_billiard_length_reference(
    table: Polytope,
    geometry: Polytope,
    *,
    max_bounces: int | None = None,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> float:
    fan = build_normal_fan(table, atol=atol)
    if fan.vertex_count == 0:
        raise ValueError("Normal fan construction yielded no vertices.")

    geometry_poly = _to_geometry_polytope(geometry, name="modern-geometry")
    combinatorics_geometry = polytope_combinatorics(geometry_poly, atol=atol)
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


def minkowski_billiard_length_fast(
    table: Polytope,
    geometry: Polytope,
    *,
    max_bounces: int | None = None,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> float:
    fan = build_normal_fan(table, atol=atol)
    if fan.vertex_count == 0:
        raise ValueError("Normal fan construction yielded no vertices.")

    geometry_poly = _to_geometry_polytope(geometry, name="modern-geometry")
    product_result = _try_product_decomposition(
        fan.polytope,
        geometry_poly,
        max_bounces=max_bounces,
        atol=atol,
    )
    if product_result is not None:
        return product_result

    geometry_vertices = polytope_combinatorics(geometry_poly, atol=atol).vertices
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
        completion_bounds = _completion_bounds(start, neighbors, length_matrix, max_length=max_length)
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
    candidates: list[tuple[int, ...]] = []
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
        segment_length = _support_function(geometry_vertices, displacement)
        total += float(segment_length)
    return total


def _pairwise_lengths(
    fan: MinkowskiNormalFan,
    geometry_vertices: Float[Array, " num_vertices_t dimension"],
) -> tuple[list[list[float]], float]:
    num_vertices = fan.vertex_count
    length_matrix = [[math.inf for _ in range(num_vertices)] for _ in range(num_vertices)]
    min_edge = math.inf
    for origin in range(num_vertices):
        for destination in fan.neighbors[origin]:
            displacement = fan.vertices[destination] - fan.vertices[origin]
            segment_length = float(_support_function(geometry_vertices, displacement))
            length_matrix[origin][destination] = segment_length
            if segment_length < min_edge:
                min_edge = segment_length
    return length_matrix, min_edge


def _dfs_prefix(
    *,
    start: int,
    current: int,
    visited_mask: int,
    steps_remaining: int,
    current_length: float,
    neighbors: tuple[tuple[int, ...], ...],
    length_matrix: list[list[float]],
    min_edge_length: float,
    completion_bounds: list[list[float]],
    best_overall_ref: list[float],
    prefix_best: dict[tuple[int, int, int, int], float],
) -> None:
    key = (start, current, visited_mask, steps_remaining)
    best_length = prefix_best.get(key)
    if best_length is not None and current_length >= best_length - 1e-12:
        return
    prefix_best[key] = current_length

    best_overall = best_overall_ref[0]
    if steps_remaining == 0:
        closing = length_matrix[current][start]
        if math.isfinite(closing):
            total = current_length + closing
            if total < best_overall:
                best_overall_ref[0] = total
        return

    for neighbor in neighbors[current]:
        if visited_mask & (1 << neighbor):
            continue
        segment = length_matrix[current][neighbor]
        if not math.isfinite(segment):
            continue
        remaining = steps_remaining - 1
        if remaining == 0:
            completion = length_matrix[neighbor][start]
        else:
            completion = completion_bounds[neighbor][remaining]
        if not math.isfinite(completion):
            continue
        lower_bound = current_length + segment + completion
        if remaining > 0:
            lower_bound = min(
                lower_bound,
                current_length + segment + (remaining + 1) * min_edge_length,
            )
        if lower_bound >= best_overall_ref[0]:
            continue
        _dfs_prefix(
            start=start,
            current=neighbor,
            visited_mask=visited_mask | (1 << neighbor),
            steps_remaining=remaining,
            current_length=current_length + segment,
            neighbors=neighbors,
            length_matrix=length_matrix,
            min_edge_length=min_edge_length,
            completion_bounds=completion_bounds,
            best_overall_ref=best_overall_ref,
            prefix_best=prefix_best,
        )


def _completion_bounds(
    start: int,
    neighbors: tuple[tuple[int, ...], ...],
    length_matrix: list[list[float]],
    *,
    max_length: int,
) -> list[list[float]]:
    num_vertices = len(neighbors)
    max_remaining = max(0, max_length - 2)
    bounds = [[math.inf for _ in range(max_remaining + 1)] for _ in range(num_vertices)]
    for vertex in range(num_vertices):
        bounds[vertex][0] = length_matrix[vertex][start]
    for remaining in range(1, max_remaining + 1):
        for vertex in range(num_vertices):
            best = math.inf
            for neighbor in neighbors[vertex]:
                if neighbor == start:
                    continue
                segment = length_matrix[vertex][neighbor]
                if not math.isfinite(segment):
                    continue
                completion = bounds[neighbor][remaining - 1]
                if not math.isfinite(completion):
                    continue
                candidate = segment + completion
                if candidate < best:
                    best = candidate
            bounds[vertex][remaining] = best
    return bounds


def _pair_to_polytope(
    polytope: _GeometryPolytope,
    blocks: tuple[tuple[int, ...], ...],
) -> list[_GeometryPolytope] | None:
    B, c = polytope.halfspace_data()
    assignments: list[list[int]] = [[] for _ in blocks]
    for row_index, row in enumerate(B):
        support = {int(idx) for idx in jnp.where(jnp.abs(row) > 1e-12)[0].tolist()}
        matched = False
        for block_index, block in enumerate(blocks):
            block_set = set(block)
            if support.issubset(block_set):
                assignments[block_index].append(row_index)
                matched = True
                break
        if not matched:
            return None
    result: list[_GeometryPolytope] = []
    for block_index, (indices, block) in enumerate(zip(assignments, blocks)):
        if not indices:
            return None
        row_indices = jnp.asarray(tuple(indices), dtype=jnp.int64)
        column_indices = jnp.asarray(tuple(block), dtype=jnp.int64)
        sub_B = jnp.take(B, row_indices, axis=0)
        sub_B = jnp.take(sub_B, column_indices, axis=1)
        sub_c = jnp.take(c, row_indices, axis=0)
        result.append(
            _GeometryPolytope(
                name=f"{polytope.name}-block-{block_index}",
                B=sub_B,
                c=sub_c,
                description=f"Coordinate block extracted from {polytope.name}.",
            )
        )
    return result


def _try_product_decomposition(
    billiard_table: _GeometryPolytope,
    geometry: _GeometryPolytope,
    *,
    max_bounces: int | None,
    atol: float,
) -> float | None:
    blocks_table = _coordinate_blocks(billiard_table.B, tol=1e-12)
    blocks_geometry = _coordinate_blocks(geometry.B, tol=1e-12)
    if len(blocks_table) <= 1 or len(blocks_geometry) <= 1:
        return None
    if any(len(block) < 2 for block in blocks_table):
        return None
    if any(len(block) < 2 for block in blocks_geometry):
        return None
    if len(blocks_table) != len(blocks_geometry):
        return None
    for block_table, block_geometry in zip(blocks_table, blocks_geometry):
        if block_table != block_geometry:
            return None
    sub_tables = _pair_to_polytope(billiard_table, blocks_table)
    sub_geometry = _pair_to_polytope(geometry, blocks_geometry)
    if sub_tables is None or sub_geometry is None:
        return None
    if len(sub_tables) != len(sub_geometry):
        return None
    total = 0.0
    for sub_table, sub_geom in zip(sub_tables, sub_geometry):
        block_max = None
        if max_bounces is not None:
            block_max = max(3, min(max_bounces, sub_table.dimension + 2))
        total += minkowski_billiard_length_fast(
            Polytope(
                normals=sub_table.B,
                offsets=sub_table.c,
                vertices=jnp.empty((0, sub_table.dimension), dtype=jnp.float64),
                incidence=jnp.empty((0, sub_table.facets), dtype=bool),
            ),
            Polytope(
                normals=sub_geom.B,
                offsets=sub_geom.c,
                vertices=jnp.empty((0, sub_geom.dimension), dtype=jnp.float64),
                incidence=jnp.empty((0, sub_geom.facets), dtype=bool),
            ),
            max_bounces=block_max,
            atol=atol,
        )
    return total


__all__ = [
    "build_normal_fan",
    "minkowski_billiard_length_reference",
    "minkowski_billiard_length_fast",
]
