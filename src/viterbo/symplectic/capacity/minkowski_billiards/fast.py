"""Optimised Minkowski billiard solver with memoised search."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.polytopes import Polytope, polytope_combinatorics
from viterbo.symplectic.capacity.minkowski_billiards.fan import (
    MinkowskiNormalFan,
    build_normal_fan,
    coordinate_partition,
)
from viterbo.symplectic.core import support_function


def compute_minkowski_billiard_length_fast(
    billiard_table: Polytope,
    geometry: Polytope,
    *,
    max_bounces: int | None = None,
    atol: float = 1e-9,
) -> float:
    """Return the minimal closed path length using dynamic programming."""

    fan = build_normal_fan(billiard_table, atol=atol)
    if fan.vertex_count == 0:
        raise ValueError("Normal fan construction yielded no vertices.")

    product_result = _try_product_decomposition(
        billiard_table,
        geometry,
        max_bounces=max_bounces,
        atol=atol,
    )
    if product_result is not None:
        return product_result

    geometry_vertices = polytope_combinatorics(geometry, atol=atol).vertices
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
            start,
            neighbors,
            length_matrix,
            max_length=max_length,
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
            segment_length = float(support_function(geometry_vertices, displacement))
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
            steps_remaining=steps_remaining - 1,
            current_length=current_length + segment,
            neighbors=neighbors,
            length_matrix=length_matrix,
            min_edge_length=min_edge_length,
            completion_bounds=completion_bounds,
            best_overall_ref=best_overall_ref,
            prefix_best=prefix_best,
        )


def _try_product_decomposition(
    billiard_table: Polytope,
    geometry: Polytope,
    *,
    max_bounces: int | None,
    atol: float,
) -> float | None:
    blocks_table = coordinate_partition(billiard_table)
    blocks_geometry = coordinate_partition(geometry)

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

    sub_tables = _split_polytope(billiard_table, blocks_table)
    sub_geometry = _split_polytope(geometry, blocks_geometry)

    if sub_tables is None or sub_geometry is None:
        return None

    if len(sub_tables) != len(sub_geometry):
        return None

    total = 0.0
    for index, (sub_table, sub_geom) in enumerate(zip(sub_tables, sub_geometry)):
        block_max = None
        if max_bounces is not None:
            block_max = max(3, min(max_bounces, sub_table.dimension + 2))
        total += compute_minkowski_billiard_length_fast(
            sub_table,
            sub_geom,
            max_bounces=block_max,
            atol=atol,
        )
    return total


def _split_polytope(
    polytope: Polytope,
    blocks: tuple[tuple[int, ...], ...],
) -> list[Polytope] | None:
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

    result: list[Polytope] = []
    for block_index, (indices, block) in enumerate(zip(assignments, blocks)):
        if not indices:
            return None
        row_indices = tuple(indices)
        column_indices = tuple(block)
        row_array = jnp.asarray(row_indices, dtype=jnp.int64)
        column_array = jnp.asarray(column_indices, dtype=jnp.int64)
        sub_B = jnp.take(B, row_array, axis=0)
        sub_B = jnp.take(sub_B, column_array, axis=1)
        sub_c = jnp.take(c, row_array, axis=0)
        result.append(
            Polytope(
                name=f"{polytope.name}-block-{block_index}",
                B=sub_B,
                c=sub_c,
                description=f"Coordinate block extracted from {polytope.name}.",
            )
        )

    return result


def _completion_bounds(
    start: int,
    neighbors: tuple[tuple[int, ...], ...],
    length_matrix: list[list[float]],
    *,
    max_length: int,
) -> list[list[float]]:
    """Precompute lower bounds for returning to ``start`` with fixed edge budgets."""

    num_vertices = len(neighbors)
    max_remaining = max(0, max_length - 2)
    bounds = [
        [math.inf for _ in range(max_remaining + 1)] for _ in range(num_vertices)
    ]

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
