"""Shared utilities for half-space representations."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float


def validate_halfspace_data(
    B: Float[np.ndarray, "num_facets dimension"],
    c: Float[np.ndarray, "num_facets"],
) -> tuple[Float[np.ndarray, "num_facets dimension"], Float[np.ndarray, "num_facets"]]:
    """Validate and normalize half-space inputs."""
    matrix = np.asarray(B, dtype=float)
    offsets = np.asarray(c, dtype=float)

    if matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        msg = "Offset vector c must match the number of facets."
        raise ValueError(msg)

    return matrix, offsets


def unique_rows(
    points: Float[np.ndarray, "num_points dimension"],
    *,
    atol: float,
) -> Float[np.ndarray, "num_unique dimension"]:
    """Deduplicate stacked vectors using an infinity-norm tolerance."""
    if points.size == 0:
        return points

    order = np.lexsort(points.T)
    keep_indices: list[int] = []
    for index in order:
        candidate = points[index]
        if not keep_indices:
            keep_indices.append(index)
            continue
        previous = points[keep_indices[-1]]
        if np.all(np.abs(candidate - previous) <= atol):
            continue
        keep_indices.append(index)
    return points[np.array(keep_indices, dtype=int)]


def deduplicate_facets(
    matrix: Float[np.ndarray, "num_facets dimension"],
    offsets: Float[np.ndarray, "num_facets"],
    *,
    atol: float,
) -> tuple[
    Float[np.ndarray, "num_unique_facets dimension"], Float[np.ndarray, "num_unique_facets"]
]:
    """Remove near-duplicate facet rows with shared offsets."""
    keep: list[int] = []
    for index, row in enumerate(matrix):
        duplicate = False
        for keep_index in keep:
            if (
                np.all(np.abs(row - matrix[keep_index]) <= atol)
                and abs(offsets[index] - offsets[keep_index]) <= atol
            ):
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    return matrix[keep, :], offsets[keep]
