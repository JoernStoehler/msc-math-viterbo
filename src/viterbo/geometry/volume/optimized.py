"""Optimised Euclidean volume estimators."""

from __future__ import annotations

import numpy as np
import scipy.spatial as _spatial  # type: ignore[reportMissingTypeStubs]
from jaxtyping import Float

from viterbo.geometry.halfspaces import optimized as halfspaces_optimized
from viterbo.geometry.volume import _shared

ConvexHull = _spatial.ConvexHull
Delaunay = _spatial.Delaunay
QhullError = _spatial.QhullError


def polytope_volume(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Optimised volume via Delaunay triangulation, with hull fallback."""
    vertices = halfspaces_optimized.enumerate_vertices(B, c, atol=atol)
    try:
        triangulation = Delaunay(vertices, qhull_options="QJ")
    except QhullError:
        hull = ConvexHull(vertices, qhull_options="QJ")
        return float(hull.volume)
    simplices = vertices[triangulation.simplices]
    return _shared.volume_of_simplices(simplices)
