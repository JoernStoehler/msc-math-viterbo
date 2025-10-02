"""
Numerical building blocks for experimenting with the Viterbo conjecture.

This module follows Google docstring style and explicit jaxtyping shapes.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from jaxtyping import Float

Vector = Float[np.ndarray, " n"]
ZERO_TOLERANCE: Final[float] = 1e-12


def standard_symplectic_matrix(dimension: int) -> Float[np.ndarray, " d d"]:
    r"""
    Return the standard symplectic matrix on R^d.

    Args:
      dimension: Even integer >= 2. The resulting matrix encodes the bilinear
        form ``ω(x, y) = x^T J y`` with block structure ``J = [[0, I_n], [-I_n, 0]]``
        for ``n = d / 2``.

    Returns:
      The ``d × d`` symplectic matrix ``J``.

    Raises:
      ValueError: If ``dimension`` is odd or smaller than 2.

    """
    if dimension % 2 != 0 or dimension < 2:
        msg = "The standard symplectic matrix is defined for even d >= 2."
        raise ValueError(msg)

    half = dimension // 2
    upper = np.hstack((np.zeros((half, half)), np.eye(half)))
    lower = np.hstack((-np.eye(half), np.zeros((half, half))))
    matrix: Float[np.ndarray, "d d"] = np.vstack((upper, lower))
    return matrix


def symplectic_product(
    first: Vector,
    second: Vector,
    *,
    matrix: Float[np.ndarray, " d d"] | None = None,
) -> float:
    r"""
    Evaluate the symplectic form of two vectors.

    Args:
      first: Vector in ``R^{2n}``.
      second: Vector in ``R^{2n}``.
      matrix: Optional symplectic matrix ``J``. Defaults to the standard form
        for ``dimension = len(first)``.

    Returns:
      The scalar ``first^T J second``.

    Raises:
      ValueError: If inputs are not one-dimensional, dimensions differ, or
        ``matrix`` does not match the vector dimension.

    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if first.ndim != 1 or second.ndim != 1:
        msg = "Symplectic product expects one-dimensional input vectors."
        raise ValueError(msg)

    if first.shape[0] != second.shape[0]:
        msg = "Vectors must have the same dimension."
        raise ValueError(msg)

    dimension = first.shape[0]
    if matrix is None:
        matrix = standard_symplectic_matrix(dimension)
    else:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (dimension, dimension):
            msg = "Symplectic matrix must match the vector dimension."
            raise ValueError(msg)

    value = float(first @ matrix @ second)
    return value


def support_function(
    vertices: Float[np.ndarray, " num_vertices d"],
    direction: Vector,
) -> float:
    r"""
    Evaluate the support function of a convex body from its vertices.

    The support function of ``K`` in direction ``u`` is
    ``h_K(u) = sup_{x in K} <u, x>``. For a finite vertex set the supremum
    is attained by a vertex.

    Args:
      vertices: Vertex coordinates, shape ``(num_vertices, dimension)``.
      direction: Direction vector in ``R^d``.

    Returns:
      Support value ``h_K(direction)``.

    Raises:
      ValueError: If vertices are empty, or dimensions do not match, or
        direction is not one-dimensional.

    """
    vertices = np.asarray(vertices, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if vertices.ndim != 2 or vertices.shape[0] == 0:
        msg = "Support function requires a non-empty set of vertices."
        raise ValueError(msg)

    if direction.ndim != 1:
        msg = "Direction must be a one-dimensional vector."
        raise ValueError(msg)

    if vertices.shape[1] != direction.shape[0]:
        msg = "Vertices and direction must share the same dimension."
        raise ValueError(msg)

    value = float(np.max(vertices @ direction))
    return value


def minkowski_sum(
    first_vertices: Float[np.ndarray, " m d"],
    second_vertices: Float[np.ndarray, " n d"],
) -> Float[np.ndarray, " mn d"]:
    r"""
    Return vertices of the Minkowski sum ``A + B``.

    Args:
      first_vertices: Vertex array for polytope ``A``, shape ``(m, d)``.
      second_vertices: Vertex array for polytope ``B``, shape ``(n, d)``.

    Returns:
      Array of shape ``(m * n, d)`` with all pairwise sums. No deduplication
      is performed; callers may prune or convexify later.

    Raises:
      ValueError: If an input is empty or dimensions differ.

    """
    first_vertices = np.asarray(first_vertices, dtype=float)
    second_vertices = np.asarray(second_vertices, dtype=float)

    if first_vertices.ndim != 2 or first_vertices.shape[0] == 0:
        msg = "First vertex array must be two-dimensional and non-empty."
        raise ValueError(msg)

    if second_vertices.ndim != 2 or second_vertices.shape[0] == 0:
        msg = "Second vertex array must be two-dimensional and non-empty."
        raise ValueError(msg)

    if first_vertices.shape[1] != second_vertices.shape[1]:
        msg = "Vertex arrays must share the same ambient dimension."
        raise ValueError(msg)

    sums = first_vertices[:, None, :] + second_vertices[None, :, :]
    result: Float[np.ndarray, "mn d"] = sums.reshape(-1, first_vertices.shape[1])
    return result


def normalize_vector(vector: Vector) -> Vector:
    r"""
    Return a unit vector pointing in the same direction.

    Args:
      vector: One-dimensional vector to normalize.
    Args:
      vector: One-dimensional vector to normalize.

    Returns:
      A unit vector with the same direction as ``vector``.

    Raises:
      ValueError: If ``vector`` has near-zero magnitude.

    Examples:
      >>> import numpy as np
      >>> from viterbo import normalize_vector
      >>> normalize_vector(np.array([3.0, 4.0]))
      array([0.6, 0.8])

    """
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        msg = "normalize_vector expects a one-dimensional input vector."
        raise ValueError(msg)

    norm: float = float(np.linalg.norm(vector))
    if norm <= ZERO_TOLERANCE:
        raise ValueError("Cannot normalize a vector with near-zero magnitude.")

    normalized: Vector = vector / norm  # shape: (n,)
    return normalized
