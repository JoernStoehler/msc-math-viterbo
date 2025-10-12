"""Polytope construction helpers for dataset adapters."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Bool

from viterbo._wrapped import spatial as _spatial
from viterbo.math.numerics import (
    INCIDENCE_ABS_TOLERANCE,
    INCIDENCE_REL_TOLERANCE,
)
from viterbo.datasets.types import Polytope


def build_from_halfspaces(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
) -> Polytope:
    """Return a bundle populated from half-space data."""
    normals = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    try:
        verts_np = _spatial.halfspace_intersection_vertices(normals, offsets)
        verts = jnp.asarray(verts_np, dtype=jnp.float64)
        if verts.shape[0] == 0:
            raise ValueError("no vertices")
        inc = incidence_matrix(normals, offsets, verts)
        return Polytope(normals=normals, offsets=offsets, vertices=verts, incidence=inc)
    except (ValueError, RuntimeError):
        _, dimension = normals.shape
        vertices = jnp.empty((0, dimension), dtype=jnp.float64)
        inc = incidence_matrix(normals, offsets, vertices)
        return Polytope(normals=normals, offsets=offsets, vertices=vertices, incidence=inc)


def build_from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
) -> Polytope:
    """Return a bundle populated from vertex data."""
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    eq = _spatial.convex_hull_equations(verts)
    A = jnp.asarray(eq[:, :-1], dtype=jnp.float64)
    b = jnp.asarray(eq[:, -1], dtype=jnp.float64)
    normals = A
    offsets = -b
    hull_indices_np = _spatial.convex_hull_vertices(verts)
    hull_indices = jnp.asarray(hull_indices_np, dtype=jnp.int32)
    verts_hull = verts[hull_indices]
    incidence = incidence_matrix(normals, offsets, verts_hull)
    return Polytope(normals=normals, offsets=offsets, vertices=verts_hull, incidence=incidence)


def incidence_matrix(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    vertices: Float[Array, " num_vertices dimension"],
    rtol: float = INCIDENCE_REL_TOLERANCE,
    atol: float = INCIDENCE_ABS_TOLERANCE,
) -> Bool[Array, " num_vertices num_facets"]:
    """Return vertex–facet incidence mask under ``rtol``/``atol`` tolerances."""
    return jnp.isclose((vertices @ normals.T) - offsets[None, :], 0.0, rtol=rtol, atol=atol)


def pad_polytope_bundle(
    bundle: Polytope,
    *,
    target_facets: int,
    target_vertices: int,
) -> Polytope:
    """Not implemented: reserved for future padding adapters used by datasets."""
    raise NotImplementedError("Padding is not implemented in the modern API.")


__all__ = [
    "build_from_halfspaces",
    "build_from_vertices",
    "incidence_matrix",
    "pad_polytope_bundle",
]
