"""JAX-compatible volume estimators."""

from __future__ import annotations

import os

import numpy as np
import scipy.spatial as _spatial  # type: ignore[reportMissingTypeStubs]
from jaxtyping import Float

from viterbo.geometry.halfspaces import jax as halfspaces_jax
from viterbo.geometry.volume import _shared

ConvexHull = _spatial.ConvexHull
Delaunay = _spatial.Delaunay
QhullError = _spatial.QhullError

os.environ.setdefault("JAX_ENABLE_X64", "1")


def polytope_volume(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Estimate volume using JAX-powered vertex solves with SciPy triangulation."""
    vertices = halfspaces_jax.enumerate_vertices(B, c, atol=atol)
    vertices_np = np.asarray(vertices, dtype=float)

    if vertices_np.size == 0:
        msg = "No vertices found; polytope may be empty or unbounded."
        raise ValueError(msg)

    try:
        triangulation = Delaunay(vertices_np, qhull_options="QJ")
    except QhullError:
        hull = ConvexHull(vertices_np, qhull_options="QJ")
        return float(hull.volume)

    simplices = vertices_np[triangulation.simplices]
    return _shared.volume_of_simplices(simplices)
