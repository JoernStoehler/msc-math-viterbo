"""Thin wrappers for SciPy spatial routines with explicit JAX↔NumPy conversion.

Centralizes all SciPy/NumPy usage for spatial operations so that the rest of
the codebase can remain JAX-first. Functions accept array-like inputs and
perform conversions internally.
"""

from __future__ import annotations

from typing import Any

import numpy as _np
import scipy.spatial as _spatial  # type: ignore[reportMissingTypeStubs]

# Re-export the Qhull error type for callers that need to handle failures.
from scipy.spatial import QhullError as QhullError  # type: ignore[reportMissingTypeStubs]


def convex_hull_volume(points: Any, *, qhull_options: str | None = "QJ") -> float:
    """Return the volume of the convex hull of ``points`` via Qhull."""
    pts = _np.asarray(points, dtype=float)
    hull = _spatial.ConvexHull(pts, qhull_options=qhull_options)
    return float(hull.volume)


def convex_hull_equations(points: Any, *, qhull_options: str | None = None) -> _np.ndarray:
    """Return hull equations ``[normals | offsets]`` from Qhull as a NumPy array."""
    pts = _np.asarray(points, dtype=float)
    hull = _spatial.ConvexHull(pts, qhull_options=qhull_options)
    return hull.equations


def delaunay_simplices(points: Any, *, qhull_options: str | None = "QJ") -> _np.ndarray:
    """Return Delaunay triangulation simplices as a NumPy index array."""
    pts = _np.asarray(points, dtype=float)
    tri = _spatial.Delaunay(pts, qhull_options=qhull_options)
    return tri.simplices
