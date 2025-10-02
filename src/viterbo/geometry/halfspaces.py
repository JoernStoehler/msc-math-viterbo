"""Utility helpers for polytopes represented as half-space intersections."""

from __future__ import annotations

import itertools
from typing import Final

import numpy as np
from jaxtyping import Float

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_FACET_MATRIX_AXES: Final[str] = f"{_FACET_AXIS} {_DIMENSION_AXIS}"
_POINT_MATRIX_AXES: Final[str] = "num_points dimension"
_UNIQUE_POINT_AXES: Final[str] = "num_unique dimension"
_UNIQUE_FACET_MATRIX_AXES: Final[str] = "num_unique_facets dimension"
_UNIQUE_FACET_AXIS: Final[str] = "num_unique_facets"
_VERTEX_MATRIX_AXES: Final[str] = "num_vertices dimension"


def _validate_halfspace_data(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
) -> tuple[
    Float[np.ndarray, _FACET_MATRIX_AXES],
    Float[np.ndarray, _FACET_AXIS],
]:
    """Return normalized ``(B, c)`` arrays and validate basic shape constraints."""
    matrix = np.asarray(B, dtype=float)
    offsets = np.asarray(c, dtype=float)

    if matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        msg = "Offset vector c must match the number of facets."
        raise ValueError(msg)

    return matrix, offsets


def _unique_rows(
    points: Float[np.ndarray, _POINT_MATRIX_AXES],
    *,
    atol: float,
) -> Float[np.ndarray, _UNIQUE_POINT_AXES]:
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


def _deduplicate_facets(
    matrix: Float[np.ndarray, _FACET_MATRIX_AXES],
    offsets: Float[np.ndarray, _FACET_AXIS],
    *,
    atol: float,
) -> tuple[
    Float[np.ndarray, _UNIQUE_FACET_MATRIX_AXES],
    Float[np.ndarray, _UNIQUE_FACET_AXIS],
]:
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


def enumerate_vertices(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    atol: float = 1e-9,
) -> Float[np.ndarray, _VERTEX_MATRIX_AXES]:
    r"""
    Enumerate the vertices of a bounded polytope ``{x | Bx \le c}``.

    The routine brute-forces all combinations of ``dimension`` facets, making it
    robust for the low-dimensional polytopes used in the project. Degenerate
    facet combinations are skipped via rank checks, and feasibility is verified
    with a uniform tolerance ``atol``.

    Parameters
    ----------
    B, c:
        Half-space description of the polytope.
    atol:
        Numerical tolerance used when checking feasibility and deduplicating
        vertices.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_vertices, dimension)`` listing the unique
        vertices.

    Raises
    ------
    ValueError
        If the input does not describe a bounded, full-dimensional polytope.

    """
    matrix, offsets = _validate_halfspace_data(B, c)
    num_facets, dimension = matrix.shape

    if dimension == 0:
        raise ValueError("Polytope dimension must be positive.")

    vertices: list[Float[np.ndarray, _DIMENSION_AXIS]] = []
    for indices in itertools.combinations(range(num_facets), dimension):
        subset = matrix[indices, :]
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
        raise ValueError("No vertices found; polytope may be empty or unbounded.")

    stacked = np.vstack(vertices)
    return _unique_rows(stacked, atol=atol)


def remove_redundant_facets(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    atol: float = 1e-9,
) -> tuple[
    Float[np.ndarray, _FACET_MATRIX_AXES],
    Float[np.ndarray, _FACET_AXIS],
]:
    """
    Prune redundant inequalities from a half-space description.

    A facet is considered redundant if it is not active on any vertex of the
    polytope. Vertex enumeration reuses :func:`enumerate_vertices`, so the same
    assumptions (bounded, full-dimensional polytope) apply.

    Parameters
    ----------
    B, c:
        Half-space description.
    atol:
        Numerical tolerance used to detect active facets.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Reduced ``(B, c)`` pair with redundant facets removed.

    """
    matrix, offsets = _validate_halfspace_data(B, c)
    matrix, offsets = _deduplicate_facets(matrix, offsets, atol=atol)
    vertices = enumerate_vertices(matrix, offsets, atol=atol)

    keep: list[int] = []
    for index, row in enumerate(matrix):
        distances = np.abs(vertices @ row - offsets[index])
        if np.any(distances <= atol):
            keep.append(index)

    if not keep:
        msg = "All facets were marked redundant; check the input polytope."
        raise ValueError(msg)

    reduced_B = matrix[keep, :]
    reduced_c = offsets[keep]
    return reduced_B, reduced_c


__all__ = ["enumerate_vertices", "remove_redundant_facets"]
