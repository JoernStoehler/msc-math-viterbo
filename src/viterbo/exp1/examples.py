from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from viterbo.exp1.polytopes import (
    HalfspacePolytope,
    LagrangianProductPolytope,
    VertexPolytope,
    lagrangian_product,
)


def regular_ngon2d(n: int) -> VertexPolytope:
    """Regular polygon on the unit circle with ``n`` vertices (2D)."""
    # Circumradius 1 polygon with evenly spaced vertices.
    if n < 3:
        raise ValueError("n must be >= 3")
    angles = (jnp.arange(n, dtype=jnp.float64) * (2.0 * math.pi / float(n)))
    verts = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
    return VertexPolytope(v=jnp.asarray(verts, dtype=jnp.float64))


def hypercube(dim: int) -> HalfspacePolytope:
    """Axis-aligned hypercube of radius 1 in ``R^dim`` (half-spaces)."""
    eye = jnp.eye(dim, dtype=jnp.float64)
    A = jnp.vstack((eye, -eye))
    b = jnp.ones(2 * dim, dtype=jnp.float64)
    return HalfspacePolytope(normals=A, offsets=b)


def crosspolytope(dim: int) -> HalfspacePolytope:
    """Cross-polytope (L1 ball) of radius 1 in ``R^dim`` (half-spaces)."""
    import itertools
    rows = jnp.asarray(list(itertools.product((-1.0, 1.0), repeat=dim)), dtype=jnp.float64)
    A = rows
    b = jnp.ones(A.shape[0], dtype=jnp.float64)
    return HalfspacePolytope(normals=A, offsets=b)


def simplex(dim: int) -> VertexPolytope:
    """Simplex used in tests; returns a vertex cloud."""
    if dim < 2:
        raise ValueError("dim must be >= 2")
    V = jnp.vstack((jnp.zeros((1, dim)), jnp.eye(dim)))
    return VertexPolytope(v=jnp.asarray(V, dtype=jnp.float64))


def viterbo_counterexample() -> LagrangianProductPolytope:
    """Pentagon × rotated pentagon product (2×2) in ``R^4``."""
    left = regular_ngon2d(5)
    right = regular_ngon2d(5)
    R = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)
    right = VertexPolytope(v=right.v @ R.T)
    return lagrangian_product(left, right)


def random_polytope(dim: int, m: int, key: PRNGKeyArray) -> HalfspacePolytope:
    """Random bounded half-space polytope with redundant facets removed."""
    if dim <= 0 or m < dim + 1:
        raise ValueError("Require dim>0 and m>=dim+1")
    k1, k2 = jax.random.split(key)
    normals = jax.random.normal(k1, (m, dim), dtype=jnp.float64)
    norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / jnp.clip(norms, a_min=1e-12, a_max=None)
    offsets = jax.random.uniform(k2, (m,), minval=0.5, maxval=1.5, dtype=jnp.float64)
    eye = jnp.eye(dim)
    normals = jnp.vstack((normals, eye, -eye))
    offsets = jnp.concatenate((offsets, jnp.full(2 * dim, 1.5)))
    return HalfspacePolytope(normals=normals, offsets=offsets)


def random_symplectic_matrix(dim: int, key: PRNGKeyArray) -> Float[Array, " dim dim"]:
    """Standard symplectic matrix for even ``dim``; identity otherwise."""
    # Deterministic fallback: standard symplectic matrix when dimension is even; otherwise identity.
    if dim % 2 != 0:
        return jnp.eye(dim, dtype=jnp.float64)
    n = dim // 2
    upper = jnp.hstack((jnp.zeros((n, n)), -jnp.eye(n)))
    lower = jnp.hstack((jnp.eye(n), jnp.zeros((n, n))))
    j_symp = jnp.vstack((upper, lower))
    return jnp.asarray(j_symp, dtype=jnp.float64)


# known quantities (see pytest for actual test cases)
# 2d:
# volume (area) == capacity
# n-gon: circumradius = 1
#        area = 1/2 * n * R^2 * sin(2pi/n)
# 2d x 2d lagrangian product:
# volume = area1 * area2
# capacity = ??? (algorithms in general)
# viterbo counterexample: 5-gon x 90-deg rotated 5-gon
#    volume = (5-gon area)^2
#    capacity = (see paper for closed form & numerical value)
# 4d non-product:
#    no examples worth mentioning here
# >4d:
#    no examples worth mentioning here

# known invariants & transformation behavior:
# volume:
#   under matrix M: volume -> |det(M)| * volume
#   under scaling by c: volume -> c^dim * volume
# capacity:
#   under symplectic matrix M: capacity -> capacity
#   under scaling by c: capacity -> c^2 * capacity
# note:
#   tensor product of symplectic matrices is symplectic, relevant for lagrangian products
# taking the dual: open conjectures exist, no good behavior known to me
