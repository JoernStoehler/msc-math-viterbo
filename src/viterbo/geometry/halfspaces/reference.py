"""Reference implementations for half-space utilities."""

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
    """Enumerate vertices of a bounded polytope ``{x | Bx ≤ c}``."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    num_facets, dimension = matrix.shape

    if dimension == 0:
        raise ValueError("Polytope dimension must be positive.")

    vertices: list[np.ndarray] = []
    combinations_iter = cast(
        Iterable[tuple[int, ...]], itertools.combinations(range(num_facets), dimension)
    )
    for combination in combinations_iter:
        indices = tuple(combination)
        subset = matrix[list(indices), :]
        if np.linalg.matrix_rank(subset) < dimension:
            continue

        subset_offsets = offsets[list(indices)]
        try:
            solution = np.linalg.solve(subset, subset_offsets)
        except np.linalg.LinAlgError:
            continue

        if np.all(matrix @ solution <= offsets + atol):
            vertices.append(np.asarray(solution, dtype=np.float64))

    if not vertices:
        msg = "No vertices found; polytope may be empty or unbounded."
        raise ValueError(msg)

    stacked = np.vstack(vertices)
    return _shared.unique_rows(stacked, atol=atol)


def remove_redundant_facets(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> tuple[Float[np.ndarray, " num_facets dimension"], Float[np.ndarray, " num_facets"]]:
    """Prune redundant inequalities from a half-space description."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    matrix, offsets = _shared.deduplicate_facets(matrix, offsets, atol=atol)
    vertices = enumerate_vertices(matrix, offsets, atol=atol)

    keep_mask: list[bool] = []
    for row, offset in zip(matrix, offsets, strict=True):
        distances = np.abs(vertices @ row - offset)
        keep_mask.append(bool(np.any(distances <= atol)))

    if not any(keep_mask):
        msg = "All facets were marked redundant; check the input polytope."
        raise ValueError(msg)

    keep = np.array(keep_mask, dtype=bool)
    reduced_B = matrix[keep, :]
    reduced_c = offsets[keep]
    return reduced_B, reduced_c
