"""Symplectic linear algebra helpers for the modern API.

This module packages the small utilities that the legacy ``viterbo.symplectic``
namespace previously exposed so downstream callers can depend on
``viterbo``.  The implementations mirror the legacy semantics but are
expressed using the modern numerics stack (JAX-first, float64 throughout).
"""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped import linalg as _linalg

ZERO_TOLERANCE: Final[float] = 1e-12


def standard_symplectic_matrix(dimension: int) -> Float[Array, " dim dim"]:
    """Return the canonical symplectic form J in dimension ``dim=2n``."""
    if dimension % 2 != 0 or dimension < 2:
        raise ValueError("dimension must be even and >= 2")
    n = dimension // 2
    upper = jnp.hstack((jnp.zeros((n, n)), -jnp.eye(n)))
    lower = jnp.hstack((jnp.eye(n), jnp.zeros((n, n))))
    return jnp.vstack((upper, lower)).astype(jnp.float64)


def random_symplectic_matrix(
    key: jax.Array, dimension: int, *, scale: float = 0.1
) -> Float[Array, " dim dim"]:
    """Sample a random symplectic matrix ``M ∈ Sp(2n)``."""
    J = standard_symplectic_matrix(dimension)
    A = jax.random.normal(key, (dimension, dimension), dtype=jnp.float64)
    S = 0.5 * (A + A.T)
    H = J @ S * jnp.asarray(scale, dtype=jnp.float64)
    M_np = _linalg.expm(H)
    return jnp.asarray(M_np, dtype=jnp.float64)


def symplectic_product(
    first: Float[Array, " dim"],
    second: Float[Array, " dim"],
    *,
    matrix: Float[Array, " dim dim"] | None = None,
) -> float:
    """Return the bilinear pairing ``firstᵀ J second`` under ``matrix``."""

    first = jnp.asarray(first, dtype=jnp.float64)
    second = jnp.asarray(second, dtype=jnp.float64)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Symplectic product expects one-dimensional input vectors.")
    if first.shape[0] != second.shape[0]:
        raise ValueError("Vectors must share the same dimension.")
    dimension = int(first.shape[0])
    if matrix is None:
        matrix = standard_symplectic_matrix(dimension)
    else:
        matrix = jnp.asarray(matrix, dtype=jnp.float64)
        if matrix.shape != (dimension, dimension):
            raise ValueError("Symplectic matrix must match the vector dimension.")
    return float(first @ matrix @ second)


def support_function(
    vertices: Float[Array, " num_vertices dim"],
    direction: Float[Array, " dim"],
) -> float:
    """Evaluate the support function ``h_K`` for a convex body ``K``."""
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)
    if verts.ndim != 2 or verts.shape[0] == 0:
        raise ValueError("Support function requires a non-empty set of vertices.")
    if direction.ndim != 1:
        raise ValueError("Direction must be a one-dimensional vector.")
    if verts.shape[1] != direction.shape[0]:
        raise ValueError("Vertices and direction must share the same dimension.")
    return float(jnp.max(verts @ direction))


def minkowski_sum(
    first_vertices: Float[Array, " m dim"],
    second_vertices: Float[Array, " n dim"],
) -> Float[Array, " k dim"]:
    """Return vertices describing the Minkowski sum ``A + B``."""
    first = jnp.asarray(first_vertices, dtype=jnp.float64)
    second = jnp.asarray(second_vertices, dtype=jnp.float64)
    if first.ndim != 2 or first.shape[0] == 0:
        raise ValueError("First vertex array must be two-dimensional and non-empty.")
    if second.ndim != 2 or second.shape[0] == 0:
        raise ValueError("Second vertex array must be two-dimensional and non-empty.")
    if first.shape[1] != second.shape[1]:
        raise ValueError("Vertex arrays must share the same ambient dimension.")
    sums = first[:, None, :] + second[None, :, :]
    return sums.reshape(-1, first.shape[1])


def normalize_vector(vector: Float[Array, " dim"]) -> Float[Array, " dim"]:
    """Return a unit vector pointing in the same direction as ``vector``."""
    vec = jnp.asarray(vector, dtype=jnp.float64)
    if vec.ndim != 1:
        raise ValueError("normalize_vector expects a one-dimensional input vector.")
    norm = float(jnp.linalg.norm(vec))
    if norm <= ZERO_TOLERANCE:
        raise ValueError("Cannot normalize a vector with near-zero magnitude.")
    return vec / norm


__all__ = [
    "ZERO_TOLERANCE",
    "standard_symplectic_matrix",
    "random_symplectic_matrix",
    "symplectic_product",
    "support_function",
    "minkowski_sum",
    "normalize_vector",
]

