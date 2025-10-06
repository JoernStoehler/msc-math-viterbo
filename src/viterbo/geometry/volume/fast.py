"""Fast Euclidean volume estimators (speed-optimized)."""

from __future__ import annotations

from jaxtyping import Array, Float

from viterbo._wrapped.spatial import QhullError, convex_hull_volume, delaunay_simplices
from viterbo.geometry.halfspaces import fast as halfspaces_fast
from viterbo.geometry.volume import _shared


def polytope_volume(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Optimised volume via Delaunay triangulation, with hull fallback."""
    vertices = halfspaces_fast.enumerate_vertices(B, c, atol=atol)
    try:
        simplices = delaunay_simplices(vertices, qhull_options="QJ")
    except QhullError:
        return convex_hull_volume(vertices, qhull_options="QJ")
    # simplices are indices into the provided vertices; gather and compute volume
    # Conversion to NumPy happens inside _shared.volume_of_simplices as needed via jnp
    return _shared.volume_of_simplices(vertices[simplices])
