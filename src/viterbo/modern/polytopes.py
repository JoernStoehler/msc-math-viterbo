"""Polytope construction helpers for modern API.

Implements conversions between V-representation (vertices) and
H-representation (halfspaces), using SciPy interop via
``viterbo._wrapped.spatial`` where needed. Returns JAX-first arrays
with float64 dtype and a boolean incidence matrix.
"""

from __future__ import annotations

from jaxtyping import Array, Float, Bool

from viterbo.modern.types import Polytope
from viterbo._wrapped import spatial as _spatial

import jax.numpy as jnp

def build_from_halfspaces(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
) -> Polytope:
    """Return a bundle populated from half-space data.

    Robust path: enumerate polytope vertices using half-space intersection,
    then rebuild via ``build_from_vertices`` to stabilize facet normals and
    offsets (deduplication, rounding). If the region is infeasible or unbounded,
    return an empty-vertex bundle with the given halfspaces.
    """
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
        num_facets, dimension = normals.shape
        vertices = jnp.empty((0, dimension), dtype=jnp.float64)
        inc = incidence_matrix(normals, offsets, vertices)
        return Polytope(normals=normals, offsets=offsets, vertices=vertices, incidence=inc)


def build_from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
) -> Polytope:
    """Return a bundle populated from vertex data.

    Uses Qhull to compute a robust convex hull and facet equations, which
    yields outward normals and offsets. Vertices may be re-ordered and
    de-duplicated compared to inputs to improve numerical stability.
    """
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    # Compute hull equations and extract a stable set of vertices on the hull.
    eq = _spatial.convex_hull_equations(verts)
    # SciPy returns equations of the form A x + b == 0 on facets with A outward.
    # Map to Bx <= c with B = A and c = -b.
    A = jnp.asarray(eq[:, :-1], dtype=jnp.float64)
    b = jnp.asarray(eq[:, -1], dtype=jnp.float64)
    normals = A
    offsets = -b
    # Use hull vertex indices to filter and stabilize vertices
    # (these are indices into the input array).
    hull_indices_np = _spatial.convex_hull_vertices(verts)
    hull_indices = jnp.asarray(hull_indices_np, dtype=jnp.int32)
    verts_hull = verts[hull_indices]
    incidence = incidence_matrix(normals, offsets, verts_hull)
    return Polytope(normals=normals, offsets=offsets, vertices=verts_hull, incidence=incidence)


def incidence_matrix(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    vertices: Float[Array, " num_vertices dimension"],
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> Bool[Array, " num_vertices num_facets"]:
    """Compute the incidence matrix for the given polytope data."""
    # idea: just check which vertices satisfy which halfspaces
    return jnp.isclose((vertices @ normals.T) - offsets[None, :], 0.0, rtol=rtol, atol=atol)

def pad_polytope_bundle(
    bundle: Polytope,
    *,
    target_facets: int,
    target_vertices: int,
) -> Polytope:
    """Pad the bundle to fixed sizes for batching."""
    raise NotImplementedError
