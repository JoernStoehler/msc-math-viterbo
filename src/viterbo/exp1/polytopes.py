from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.exp1.halfspaces import (
    enumerate_vertices as _enumerate_vertices,
    halfspaces_from_vertices as _halfspaces_from_vertices,
)

# dim = dimension of the ambient space
# m = number of halfspaces
# k = number of vertices
# A = normal vectors of halfspaces, b = offsets of halfspaces
# v = vertices of the polytope


@dataclass
class HalfspacePolytope:
    """Convex polytope encoded by the half-space system ``Bx ≤ c``.

    Notes:
      - ``normals`` are outward facet normals (rows of ``B``)
      - ``offsets`` are the right-hand sides (``c``)
      - Row normalization is not required; redundant facets may be present.
    """

    normals: Float[Array, " m dim"]
    offsets: Float[Array, " m"]

    def as_tuple(self) -> tuple[Float[Array, " m dim"], Float[Array, " m"]]:
        return jnp.asarray(self.normals, dtype=jnp.float64), jnp.asarray(
            self.offsets, dtype=jnp.float64
        )

    @property
    def dim(self) -> int:
        return int(jnp.asarray(self.normals).shape[1])


@dataclass
class VertexPolytope:
    """Convex polytope represented by its vertex set (rows of ``v``).

    Notes:
      - The vertex list need not be ordered; duplicate points may appear before cleanup.
      - Prefer converting to half-spaces for algorithms that operate on facets.
    """

    v: Float[Array, " k dim"]

    def as_tuple(self) -> tuple[Float[Array, " k dim"]]:
        return (jnp.asarray(self.v, dtype=jnp.float64),)

    @property
    def dim(self) -> int:
        return int(jnp.asarray(self.v).shape[1])


@dataclass
class LagrangianProductPolytope:
    """Axis-aligned Cartesian product of two 2D polytopes in ``R^4``.

    Fields use (p, q) to reflect the usual ``R^{2n} = R^n_p × R^n_q`` split.
    Encodes both half-spaces (normals/offsets) and vertices for each 2D factor.
    """

    normals_p: Float[Array, " m1 2"]
    offsets_p: Float[Array, " m1"]
    verts_p: Float[Array, " k1 2"]
    normals_q: Float[Array, " m2 2"]
    offsets_q: Float[Array, " m2"]
    verts_q: Float[Array, " k2 2"]

    def as_tuple(
        self,
    ) -> tuple[
        Float[Array, " m1 2"],
        Float[Array, " m1"],
        Float[Array, " k1 2"],
        Float[Array, " m2 2"],
        Float[Array, " m2"],
        Float[Array, " k2 2"],
    ]:
        return (
            jnp.asarray(self.normals_p, dtype=jnp.float64),
            jnp.asarray(self.offsets_p, dtype=jnp.float64),
            jnp.asarray(self.verts_p, dtype=jnp.float64),
            jnp.asarray(self.normals_q, dtype=jnp.float64),
            jnp.asarray(self.offsets_q, dtype=jnp.float64),
            jnp.asarray(self.verts_q, dtype=jnp.float64),
        )


Polytope = HalfspacePolytope | VertexPolytope | LagrangianProductPolytope


def to_vertices(P: Polytope, *, atol: float = 1e-9) -> VertexPolytope:
    """Return a vertex representation for ``P``.

    Converts half-spaces via vertex enumeration; for 2×2 products, forms the
    Cartesian product of factor vertices.
    """
    if isinstance(P, VertexPolytope):
        return P
    if isinstance(P, HalfspacePolytope):
        A, b = P.as_tuple()
        verts = _enumerate_vertices(A, b, atol=atol)
        return VertexPolytope(v=jnp.asarray(verts, dtype=jnp.float64))
    if isinstance(P, LagrangianProductPolytope):
        left = jnp.asarray(P.verts_p, dtype=jnp.float64)
        right = jnp.asarray(P.verts_q, dtype=jnp.float64)
        k1 = int(left.shape[0])
        k2 = int(right.shape[0])
        verts = jnp.concatenate(
            (jnp.repeat(left, repeats=k2, axis=0), jnp.tile(right, (k1, 1))), axis=1
        )
        return VertexPolytope(v=verts)
    raise TypeError("Unsupported polytope type for to_vertices")


def to_halfspaces(P: Polytope) -> HalfspacePolytope:
    """Return a half-space representation ``Bx ≤ c`` for ``P``.

    Converts vertices via a convex-hull half-space reconstruction; for 2×2
    products embeds factor inequalities into block coordinates of ``R^4``.
    """
    if isinstance(P, HalfspacePolytope):
        return P
    if isinstance(P, VertexPolytope):
        (verts,) = P.as_tuple()
        normals, offsets = _halfspaces_from_vertices(verts)
        return HalfspacePolytope(
            normals=jnp.asarray(normals, dtype=jnp.float64),
            offsets=jnp.asarray(offsets, dtype=jnp.float64),
        )
    if isinstance(P, LagrangianProductPolytope):
        # Embed blocks into R^4 half-spaces
        normals_left = jnp.hstack((P.normals_p, jnp.zeros((P.normals_p.shape[0], 2))))
        normals_right = jnp.hstack((jnp.zeros((P.normals_q.shape[0], 2)), P.normals_q))
        normals = jnp.vstack((normals_left, normals_right))
        offsets = jnp.concatenate((P.offsets_p, P.offsets_q))
        return HalfspacePolytope(
            normals=jnp.asarray(normals, dtype=jnp.float64),
            offsets=jnp.asarray(offsets, dtype=jnp.float64),
        )
    raise TypeError("Unsupported polytope type for to_halfspaces")


def to_lagrangian_product(P: Polytope, *, atol: float = 1e-12) -> LagrangianProductPolytope:
    """Factor a 4D polytope into an axis-aligned 2×2 product, if possible.

    Raises:
      ValueError: If the system is not 4D or contains mixed-support facets that
      prevent a clean 2D×2D block decomposition.
    """
    H = to_halfspaces(P)
    normals, offsets = H.as_tuple()
    dim = int(normals.shape[1])
    if dim != 4:
        raise ValueError("to_lagrangian_product expects a 4D polytope.")
    left_mask: list[int] = []
    right_mask: list[int] = []
    for i in range(normals.shape[0]):
        row = normals[i]
        left = jnp.linalg.norm(row[:2])
        right = jnp.linalg.norm(row[2:])
        if right <= atol and left > atol:
            left_mask.append(i)
        elif left <= atol and right > atol:
            right_mask.append(i)
        else:
            raise ValueError("Polytope appears not to be an axis-aligned 2x2 product.")
    left_idx = jnp.asarray(left_mask, dtype=int)
    right_idx = jnp.asarray(right_mask, dtype=int)
    normals1 = jnp.asarray(normals[left_idx, :2], dtype=jnp.float64)
    offsets1 = jnp.asarray(offsets[left_idx], dtype=jnp.float64)
    normals2 = jnp.asarray(normals[right_idx, 2:], dtype=jnp.float64)
    offsets2 = jnp.asarray(offsets[right_idx], dtype=jnp.float64)
    v1 = _enumerate_vertices(normals1, offsets1, atol=atol)
    v2 = _enumerate_vertices(normals2, offsets2, atol=atol)
    return LagrangianProductPolytope(
        normals_p=normals1,
        offsets_p=offsets1,
        verts_p=jnp.asarray(v1, dtype=jnp.float64),
        normals_q=normals2,
        offsets_q=offsets2,
        verts_q=jnp.asarray(v2, dtype=jnp.float64),
    )


T = TypeVar("T", HalfspacePolytope, VertexPolytope)


def matmul(mat: Float[Array, " dim dim"], P: T) -> T:
    """Apply the linear map ``x ↦ Mx`` to a polytope representation.

    For half-spaces, updates to ``B M^{-1}`` per affine image of ``Bx ≤ c``.
    For vertices, applies pointwise multiplication by ``M``.
    """
    mat = jnp.asarray(mat, dtype=jnp.float64)
    if isinstance(P, HalfspacePolytope):
        if mat.shape != (P.dim, P.dim):
            raise ValueError("Matrix shape must match polytope dimension.")
        mat_inv = jnp.linalg.inv(mat)
        A_new = jnp.asarray(P.normals) @ mat_inv
        return HalfspacePolytope(
            normals=jnp.asarray(A_new, dtype=jnp.float64),
            offsets=jnp.asarray(P.offsets, dtype=jnp.float64),
        )  # type: ignore[return-value]
    if isinstance(P, VertexPolytope):
        if mat.shape != (P.dim, P.dim):
            raise ValueError("Matrix shape must match polytope dimension.")
        V_new = jnp.asarray(P.v) @ mat.T
        return VertexPolytope(v=jnp.asarray(V_new, dtype=jnp.float64))  # type: ignore[return-value]
    raise TypeError("matmul not supported for LagrangianProductPolytope")


def scale(c: Float[Array, ""], P: Polytope) -> Polytope:
    """Return the scaled polytope ``c·P`` using the natural representation rules."""
    c = jnp.asarray(c, dtype=jnp.float64)
    if isinstance(P, HalfspacePolytope):
        return HalfspacePolytope(
            normals=jnp.asarray(P.normals, dtype=jnp.float64) / c,
            offsets=jnp.asarray(P.offsets, dtype=jnp.float64),
        )
    if isinstance(P, VertexPolytope):
        return VertexPolytope(v=jnp.asarray(P.v, dtype=jnp.float64) * c)
    if isinstance(P, LagrangianProductPolytope):
        return LagrangianProductPolytope(
            normals_p=jnp.asarray(P.normals_p, dtype=jnp.float64) / c,
            offsets_p=jnp.asarray(P.offsets_p, dtype=jnp.float64),
            verts_p=jnp.asarray(P.verts_p, dtype=jnp.float64) * c,
            normals_q=jnp.asarray(P.normals_q, dtype=jnp.float64) / c,
            offsets_q=jnp.asarray(P.offsets_q, dtype=jnp.float64),
            verts_q=jnp.asarray(P.verts_q, dtype=jnp.float64) * c,
        )
    raise TypeError("Unsupported polytope type for scale")


def lagrangian_product(P1: T, P2: T) -> LagrangianProductPolytope:
    """Build the axis-aligned 2×2 Cartesian product in ``R^4`` from 2D factors."""
    H1 = to_halfspaces(P1)
    H2 = to_halfspaces(P2)
    V1 = to_vertices(P1)
    V2 = to_vertices(P2)
    return LagrangianProductPolytope(
        normals_p=jnp.asarray(H1.normals, dtype=jnp.float64),
        offsets_p=jnp.asarray(H1.offsets, dtype=jnp.float64),
        verts_p=jnp.asarray(V1.v, dtype=jnp.float64),
        normals_q=jnp.asarray(H2.normals, dtype=jnp.float64),
        offsets_q=jnp.asarray(H2.offsets, dtype=jnp.float64),
        verts_q=jnp.asarray(V2.v, dtype=jnp.float64),
    )
