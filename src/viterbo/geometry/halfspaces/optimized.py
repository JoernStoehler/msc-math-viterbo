"""Optimised half-space helpers with vectorised kernels."""

from __future__ import annotations

import itertools
from typing import Iterable, cast

import numpy as np
from jaxtyping import Float

from viterbo.geometry.halfspaces import _shared


def enumerate_vertices(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[np.ndarray, " num_vertices dimension"]:
    """Vectorised variant that caches feasibility checks."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    num_facets, dimension = matrix.shape

    if dimension == 0:
        raise ValueError("Polytope dimension must be positive.")

    feasible_vertices: list[np.ndarray] = []
    feasibility_cache: dict[tuple[int, ...], bool] = {}

    combinations_iter = cast(
        Iterable[tuple[int, ...]], itertools.combinations(range(num_facets), dimension)
    )
    for combination in combinations_iter:
        indices = tuple(combination)
        if feasibility_cache.get(indices) is False:
            continue

        subset = matrix[list(indices), :]
        if np.linalg.matrix_rank(subset) < dimension:
            feasibility_cache[indices] = False
            continue

        subset_offsets = offsets[list(indices)]
        try:
            solution = np.linalg.solve(subset, subset_offsets)
        except np.linalg.LinAlgError:
            feasibility_cache[indices] = False
            continue

        satisfied = bool(np.all(matrix @ solution <= offsets + atol))
        feasibility_cache[indices] = satisfied
        if satisfied:
            feasible_vertices.append(np.asarray(solution, dtype=np.float64))

    if not feasible_vertices:
        msg = "No vertices found; polytope may be empty or unbounded."
        raise ValueError(msg)

    stacked = np.vstack(feasible_vertices)
    return _shared.unique_rows(stacked, atol=atol)


def remove_redundant_facets(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> tuple[Float[np.ndarray, " num_facets dimension"], Float[np.ndarray, " num_facets"]]:
    """Vectorised redundancy pruning using matrix operations."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    matrix, offsets = _shared.deduplicate_facets(matrix, offsets, atol=atol)
    vertices = enumerate_vertices(matrix, offsets, atol=atol)

    # Broadcast distances for every facet against every vertex simultaneously.
    distances = np.abs(matrix @ vertices.T - offsets[:, None])
    keep = np.any(distances <= atol, axis=1)

    if not np.any(keep):
        msg = "All facets were marked redundant; check the input polytope."
        raise ValueError(msg)

    reduced_B = matrix[keep, :]
    reduced_c = offsets[keep]
    return reduced_B, reduced_c
