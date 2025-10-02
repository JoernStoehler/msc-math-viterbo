"""Numerical building blocks for experimenting with the Viterbo conjecture."""

from __future__ import annotations

from typing import Final

import numpy as np
from jaxtyping import Float

Vector = Float[np.ndarray, " n"]
ZERO_TOLERANCE: Final[float] = 1e-12


def standard_symplectic_matrix(dimension: int) -> Float[np.ndarray, " d d"]:
    r"""Return the matrix of the standard symplectic form on ``\mathbb{R}^d``.

    Parameters
    ----------
    dimension:
        An even integer at least two. The resulting matrix encodes the bilinear
        form ``\omega(x, y) = x^{\mathsf{T}} J y`` with the block structure
        ``J = [[0, I_n], [-I_n, 0]]`` for ``n = d / 2``.

    Returns
    -------
    numpy.ndarray
        The ``d \times d`` symplectic matrix ``J``.

    Raises
    ------
    ValueError
        If ``dimension`` is not an even integer greater or equal to two.
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
    r"""Evaluate the symplectic form of two vectors.

    Parameters
    ----------
    first, second:
        Vectors in ``\mathbb{R}^{2n}``.
    matrix:
        Optional symplectic matrix ``J``. When omitted, the standard form is
        used with ``dimension = len(first)``.

    Returns
    -------
    float
        The value ``first^{\mathsf{T}} J second``.

    Raises
    ------
    ValueError
        If the vectors have incompatible shapes or if ``matrix`` does not
        define a square symplectic matrix of matching dimension.
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
    r"""Evaluate the support function of a convex body given by its vertices.

    The support function of a convex body ``K`` in direction ``u`` is defined
    by ``h_K(u) = \sup_{x \in K} \langle u, x \rangle``. When ``K`` is
    described by a finite vertex set, the supremum is attained and equals the
    maximum dot product between the direction and the vertices.

    Parameters
    ----------
    vertices:
        Array of vertex coordinates with shape ``(m, d)``.
    direction:
        Direction vector in ``\mathbb{R}^d`` along which the support function
        is evaluated.

    Returns
    -------
    float
        The support value ``h_K(direction)``.

    Raises
    ------
    ValueError
        If no vertices are provided or if the dimensionalities do not match.
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
    r"""Return vertices of the Minkowski sum ``A + B`` for finite vertex sets.

    The Minkowski sum is formed by pairwise addition of the input vertex sets.
    The output is not deduplicated; downstream callers can apply convex hull
    or vertex pruning routines when necessary.

    Parameters
    ----------
    first_vertices, second_vertices:
        Arrays of shape ``(m, d)`` and ``(n, d)`` describing the vertex sets of
        two polytopes ``A`` and ``B``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(m * n, d)`` containing all pairwise sums.

    Raises
    ------
    ValueError
        If either vertex array is empty or if their dimensions do not match.
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
    """
    Return a unit vector pointing in the same direction as ``vector``.

    Parameters
    ----------
    vector:
        One-dimensional NumPy array representing the vector to normalize.

    Returns
    -------
    numpy.ndarray
        A unit vector pointing in the same direction as ``vector``.

    Raises
    ------
    ValueError
        If ``vector`` is (numerically) the zero vector.

    Examples
    --------
    >>> import numpy as np
    >>> from viterbo import normalize_vector
    >>> normalize_vector(np.array([3.0, 4.0]))
    array([0.6, 0.8])

    """
    norm: float = float(np.linalg.norm(vector))
    if norm <= ZERO_TOLERANCE:
        raise ValueError("Cannot normalize a vector with near-zero magnitude.")

    normalized: Vector = vector / norm  # shape: (n,)
    return normalized
