"""Volume estimators for the modern API (math layer)."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped import spatial as _spatial
from viterbo.math.geometry import enumerate_vertices


def volume_reference(vertices: Float[Array, " num_vertices dimension"]) -> float:
    """Return a reference volume estimate for a vertex cloud.

    - In 2D, compute the exact polygon area via the shoelace formula on the
      convex hull of the provided vertices (ordered by angle around centroid).
    - In higher dimensions, fall back to Qhull volume.
    """
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    d = int(verts.shape[1])
    if d == 2:
        order_np = _spatial.convex_hull_vertices(verts)
        P = verts[jnp.asarray(order_np, dtype=jnp.int32)]
        x = P[:, 0]
        y = P[:, 1]
        x_next = jnp.concatenate([x[1:], x[:1]], axis=0)
        y_next = jnp.concatenate([y[1:], y[:1]], axis=0)
        area = 0.5 * jnp.abs(jnp.sum(x * y_next - y * x_next))
        return float(area)
    return float(_spatial.convex_hull_volume(verts))


def volume_padded(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    method: str,
) -> Float[Array, " batch"]:
    """Compute batched volumes using a padding-friendly method.

    Placeholder for batching semantics; returns zeros of appropriate shape.
    """
    batch = normals.shape[0]
    return jnp.zeros((batch,), dtype=jnp.float64)


def _volume_of_simplices(
    simplex_vertices: Float[Array, " num_simplices vertices dimension"],
) -> float:
    v = jnp.asarray(simplex_vertices)
    base = v[:, 0, :]
    edges = v[:, 1:, :] - base[:, None, :]
    determinants = jnp.linalg.det(edges)
    dimension = v.shape[2]
    total = jnp.sum(jnp.abs(determinants)) / math.factorial(int(dimension))
    return float(total)


def polytope_volume_reference(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Return reference convex volume using hull volume over enumerated vertices."""
    vertices = enumerate_vertices(B, c, atol=atol)
    return float(_spatial.convex_hull_volume(vertices, qhull_options="QJ"))


def polytope_volume_fast(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Return fast convex volume via Delaunay simplices, fallback to hull volume."""
    vertices = enumerate_vertices(B, c, atol=atol)
    try:
        simplices = _spatial.delaunay_simplices(vertices, qhull_options="QJ")
    except _spatial.QhullError:
        return float(_spatial.convex_hull_volume(vertices, qhull_options="QJ"))
    return _volume_of_simplices(vertices[simplices])
