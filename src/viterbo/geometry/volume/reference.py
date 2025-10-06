"""Reference Euclidean volume estimators for convex polytopes (trusted)."""

from __future__ import annotations

from jaxtyping import Array, Float

from viterbo._wrapped.spatial import convex_hull_volume
from viterbo.geometry.halfspaces import reference as halfspaces_reference


def polytope_volume(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Trusted volume via SciPy's ConvexHull (Qhull)."""
    vertices = halfspaces_reference.enumerate_vertices(B, c, atol=atol)
    return convex_hull_volume(vertices, qhull_options="QJ")
