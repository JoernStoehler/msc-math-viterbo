"""Symplectic linear algebra helpers for the modern API.

Provides utilities to construct the standard symplectic form and to sample
random elements of Sp(2n) via the matrix exponential of a Hamiltonian matrix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped import linalg as _linalg


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
    """Sample a random symplectic matrix ``M ∈ Sp(2n)``.

    Construction: draw a symmetric matrix ``S`` from N(0,1), build a Hamiltonian
    matrix ``H = J S`` where ``J`` is the standard symplectic form, and return
    ``M = expm(H)``. The optional ``scale`` controls the magnitude of the draw.
    """
    J = standard_symplectic_matrix(dimension)
    A = jax.random.normal(key, (dimension, dimension), dtype=jnp.float64)
    S = 0.5 * (A + A.T)
    H = J @ S * jnp.asarray(scale, dtype=jnp.float64)
    M_np = _linalg.expm(H)
    return jnp.asarray(M_np, dtype=jnp.float64)

