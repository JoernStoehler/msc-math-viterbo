"""Euclidean volume estimators for convex polytopes."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from jaxtyping import Float
from scipy.spatial import ConvexHull, Delaunay, QhullError

from .halfspaces import enumerate_vertices

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_FACET_MATRIX_AXES: Final[str] = f"{_FACET_AXIS} {_DIMENSION_AXIS}"
_SIMPLEX_AXES: Final[str] = "num_simplices vertices dimension"


def _volume_of_simplices(
    simplex_vertices: Float[np.ndarray, _SIMPLEX_AXES],
) -> float:
    """Return the total volume of simplices defined by ``simplex_vertices``."""
    base = simplex_vertices[:, 0, :]  # shape: (k, d)
    edges = simplex_vertices[:, 1:, :] - base[:, None, :]
    determinants = np.linalg.det(edges)
    dimension = simplex_vertices.shape[2]
    return float(np.sum(np.abs(determinants)) / math.factorial(dimension))


def polytope_volume_reference(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    atol: float = 1e-9,
) -> float:
    """
    Trusted Euclidean volume computed via Qhull (SciPy ``ConvexHull``).

    Qhull [Barber1996]_ is a mature computational geometry library that powers
    SciPy's ``ConvexHull`` implementation. We rely on its volume computation as a
    correctness oracle and cross-check faster estimators against it.
    """
    vertices = enumerate_vertices(B, c, atol=atol)
    hull = ConvexHull(vertices, qhull_options="QJ")
    return float(hull.volume)


def polytope_volume_fast(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    atol: float = 1e-9,
) -> float:
    """Optimized volume estimator using Delaunay triangulation of vertices."""
    vertices = enumerate_vertices(B, c, atol=atol)
    try:
        triangulation = Delaunay(vertices, qhull_options="QJ")
    except QhullError:
        hull = ConvexHull(vertices, qhull_options="QJ")
        return float(hull.volume)
    simplices = vertices[triangulation.simplices]
    return _volume_of_simplices(simplices)


__all__ = [
    "polytope_volume_fast",
    "polytope_volume_reference",
]
