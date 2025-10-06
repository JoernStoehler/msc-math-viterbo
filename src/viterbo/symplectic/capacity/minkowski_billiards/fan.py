"""Normal fan construction utilities for Minkowski billiards."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.polytopes import (
    NormalCone,
    Polytope,
    polytope_combinatorics,
)


@dataclass(frozen=True)
class MinkowskiNormalFan:
    """Normal fan representation of a polytope."""

    polytope: Polytope
    vertices: Float[Array, " num_vertices dimension"]
    cones: tuple[NormalCone, ...]
    adjacency: Array
    neighbors: tuple[tuple[int, ...], ...]
    coordinate_blocks: tuple[tuple[int, ...], ...]

    @property
    def dimension(self) -> int:
        """Return the ambient dimension of the fan."""

        return int(self.vertices.shape[1])

    @property
    def vertex_count(self) -> int:
        """Return the number of normal cones/vertices."""

        return int(self.vertices.shape[0])


def build_normal_fan(
    polytope: Polytope,
    *,
    atol: float = 1e-9,
) -> MinkowskiNormalFan:
    """Construct the normal fan of ``polytope`` from cached combinatorics."""

    combinatorics = polytope_combinatorics(polytope, atol=atol)
    cones = combinatorics.normal_cones
    if not cones:
        raise ValueError("Polytope must have at least one vertex to build a normal fan.")
    vertices = jnp.stack([cone.vertex for cone in cones], axis=0)

    adjacency = _vertex_adjacency(cones, dimension=polytope.dimension)
    neighbors = tuple(
        tuple(int(index) for index in jnp.where(row)[0].tolist())
        for row in adjacency
    )
    coordinate_blocks = coordinate_partition(polytope)

    return MinkowskiNormalFan(
        polytope=polytope,
        vertices=vertices,
        cones=cones,
        adjacency=adjacency,
        neighbors=neighbors,
        coordinate_blocks=coordinate_blocks,
    )


def _vertex_adjacency(
    cones: Sequence[NormalCone],
    *,
    dimension: int,
) -> Array:
    """Return an adjacency matrix for the normal cones."""

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


def coordinate_partition(
    polytope: Polytope,
    *,
    tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    """Return coordinate blocks induced by facet support sets."""

    return _coordinate_blocks(polytope.B, tol=tol)


def _coordinate_blocks(
    matrix: Float[Array, " num_facets dimension"],
    *,
    tol: float,
) -> tuple[tuple[int, ...], ...]:
    """Internal helper to compute coordinate blocks for a raw matrix."""

    dimension = int(matrix.shape[1])
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
