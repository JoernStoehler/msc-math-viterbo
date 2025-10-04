"""JAX-compatible half-space helpers."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools
from typing import Iterable, Sequence, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from viterbo.geometry.halfspaces import _shared


def _solve_subset(
    matrix: np.ndarray,
    offsets: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    subset = matrix[list(indices), :].astype(np.float64, copy=False)
    subset_offsets = offsets[list(indices)].astype(np.float64, copy=False)
    return np.asarray(jnp.linalg.solve(subset, subset_offsets), dtype=float)


def enumerate_vertices(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[np.ndarray, " num_vertices dimension"]:
    """Enumerate vertices using ``jax.numpy`` linear algebra."""
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

        try:
            solution = _solve_subset(matrix, offsets, indices)
        except np.linalg.LinAlgError:
            continue

        if np.all(matrix @ solution <= offsets + atol):
            vertices.append(solution)

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
    """Prune redundant inequalities with JAX-powered solves."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    matrix, offsets = _shared.deduplicate_facets(matrix, offsets, atol=atol)
    vertices = enumerate_vertices(matrix, offsets, atol=atol)

    distances = np.abs(matrix @ vertices.T - offsets[:, None])
    keep = np.any(distances <= atol, axis=1)

    if not np.any(keep):
        msg = "All facets were marked redundant; check the input polytope."
        raise ValueError(msg)

    reduced_B = matrix[keep, :]
    reduced_c = offsets[keep]
    return reduced_B, reduced_c
