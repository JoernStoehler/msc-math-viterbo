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

from viterbo._wrapped.optimize import linprog as _linprog


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


def convex_hull_vertices(points: Any, *, qhull_options: str | None = None) -> _np.ndarray:
    """Return indices of input points that are vertices of the convex hull."""
    pts = _np.asarray(points, dtype=float)
    hull = _spatial.ConvexHull(pts, qhull_options=qhull_options)
    return hull.vertices


def delaunay_simplices(points: Any, *, qhull_options: str | None = "QJ") -> _np.ndarray:
    """Return Delaunay triangulation simplices as a NumPy index array."""
    pts = _np.asarray(points, dtype=float)
    tri = _spatial.Delaunay(pts, qhull_options=qhull_options)
    return tri.simplices


def halfspace_intersection_vertices(B: Any, c: Any, *, atol: float = 1e-12) -> _np.ndarray:
    """Enumerate vertices of the bounded polyhedron {x: Bx <= c}.

    Uses SciPy's HalfspaceIntersection. Requires a feasible interior point;
    we obtain one by solving a feasibility LP with scipy.optimize.linprog.

    Returns a NumPy array of vertices with shape (k, d). Raises on infeasible
    or unbounded inputs.
    """
    Bm = _np.asarray(B, dtype=float)
    cv = _np.asarray(c, dtype=float)
    _, d = Bm.shape
    # Compute Chebyshev center (x, r) maximizing margin r >= 0 s.t.
    # a_i^T x + ||a_i|| r <= c_i for all i. Guarantees strict interior if r>0.
    norms = _np.linalg.norm(Bm, axis=1)
    A_ext = _np.hstack([Bm, norms[:, None]])
    c_ext = _np.zeros((d + 1,), dtype=float)
    c_ext[-1] = -1.0  # maximize r -> minimize -r
    bounds = [(None, None)] * d + [(0.0, None)]
    res = _linprog(
        c=c_ext,
        A_ub=A_ext,
        b_ub=cv,
        A_eq=None,
        b_eq=None,
        bounds=bounds,
        method="highs",
    )
    if not bool(res.success):
        raise ValueError("Halfspace system is infeasible or unbounded.")
    sol = _np.asarray(res.x, dtype=float)
    interior = sol[:d]
    # SciPy expects halfspaces as [a, b] rows with a x + b <= 0
    halfspaces = _np.hstack([Bm, -cv[:, None]])
    hs = _spatial.HalfspaceIntersection(halfspaces, interior)
    verts = _np.asarray(hs.intersections, dtype=float)
    # Deduplicate approximately based on atol
    if verts.size == 0:
        return verts.reshape((0, d))
    scaled = _np.round(verts / float(atol)).astype(_np.int64)
    _, unique_idx = _np.unique(scaled, axis=0, return_index=True)
    unique = verts[_np.sort(unique_idx)]
    return unique
