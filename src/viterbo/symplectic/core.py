"""Numerical building blocks for experimenting with the Viterbo conjecture.

JAX-first: uses ``jax.numpy`` for array math and Google-style docstrings with
explicit jaxtyping shapes. All public helpers validate shapes/dtypes and fail
fast with precise exceptions.
"""

from __future__ import annotations

from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float

ZERO_TOLERANCE: Final[float] = 1e-12


def standard_symplectic_matrix(
    dimension: int,
) -> Float[Array, " dimension dimension"]:
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
    upper = jnp.hstack((jnp.zeros((half, half)), jnp.eye(half)))
    lower = jnp.hstack((-jnp.eye(half), jnp.zeros((half, half))))
    matrix: Float[Array, " dimension dimension"] = jnp.vstack((upper, lower))
    return matrix


def symplectic_product(
    first: Float[Array, " dimension"],
    second: Float[Array, " dimension"],
    *,
    matrix: Float[Array, " dimension dimension"] | None = None,
) -> float:
    r"""
    Evaluate the symplectic form of two vectors.

    Args:
      first: One-dimensional vector in ``R^{2n}``.
      second: One-dimensional vector in ``R^{2n}``.
      matrix: Optional symplectic matrix ``J``. Defaults to the standard form
        for ``dimension = len(first)``.

    Returns:
      The scalar ``first^T J second``.

    Raises:
      ValueError: If inputs are not one-dimensional, dimensions differ, or
        ``matrix`` does not match the vector dimension.

    """
    first = jnp.asarray(first, dtype=jnp.float64)
    second = jnp.asarray(second, dtype=jnp.float64)

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
        matrix = jnp.asarray(matrix, dtype=jnp.float64)
        if matrix.shape != (dimension, dimension):
            msg = "Symplectic matrix must match the vector dimension."
            raise ValueError(msg)

    value = float(first @ matrix @ second)
    return value


def support_function(
    vertices: Float[Array, " num_vertices dimension"],
    direction: Float[Array, " dimension"],
) -> float:
    r"""
    Evaluate the support function of a convex body from its vertices.

    The support function of ``K`` in direction ``u`` is
    ``h_K(u) = sup_{x in K} <u, x>``. For a finite vertex set the supremum
    is attained by a vertex.

    Args:
      vertices: Vertex coordinates, shape ``(num_vertices, dimension)``.
      direction: Direction vector in ``R^{dimension}``.

    Returns:
      Support value ``h_K(direction)``.

    Raises:
      ValueError: If vertices are empty, or dimensions do not match, or
        direction is not one-dimensional.

    """
    vertices = jnp.asarray(vertices, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)

    if vertices.ndim != 2 or vertices.shape[0] == 0:
        msg = "Support function requires a non-empty set of vertices."
        raise ValueError(msg)

    if direction.ndim != 1:
        msg = "Direction must be a one-dimensional vector."
        raise ValueError(msg)

    if vertices.shape[1] != direction.shape[0]:
        msg = "Vertices and direction must share the same dimension."
        raise ValueError(msg)

    value = float(jnp.max(vertices @ direction))
    return value


def minkowski_sum(
    first_vertices: Float[Array, " m dimension"],
    second_vertices: Float[Array, " n dimension"],
) -> Float[Array, " k dimension"]:
    r"""
    Return vertices of the Minkowski sum ``A + B``.

    Args:
      first_vertices: Vertex array for polytope ``A``, shape ``(m, dimension)``.
      second_vertices: Vertex array for polytope ``B``, shape ``(n, dimension)``.

    Returns:
      Array of shape ``(k, dimension)`` with all pairwise sums, ``k = m * n``. No deduplication
      is performed; callers may prune or convexify later.

    Raises:
      ValueError: If an input is empty or dimensions differ.

    """
    first_vertices = jnp.asarray(first_vertices, dtype=jnp.float64)
    second_vertices = jnp.asarray(second_vertices, dtype=jnp.float64)

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
    result: Float[Array, " k dimension"] = sums.reshape(-1, first_vertices.shape[1])
    return result


def normalize_vector(
    vector: Float[Array, " dimension"],
) -> Float[Array, " dimension"]:
    r"""
    Return a unit vector pointing in the same direction.

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
    vector = jnp.asarray(vector, dtype=jnp.float64)

    if vector.ndim != 1:
        msg = "normalize_vector expects a one-dimensional input vector."
        raise ValueError(msg)

    norm: float = float(jnp.linalg.norm(vector))
    if norm <= ZERO_TOLERANCE:
        raise ValueError("Cannot normalize a vector with near-zero magnitude.")

    normalized: Float[Array, " dimension"] = vector / norm
    return normalized
