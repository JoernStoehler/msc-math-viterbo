"""Reference Euclidean volume estimators for convex polytopes."""

from __future__ import annotations

import numpy as np
import scipy.spatial as _spatial  # type: ignore[reportMissingTypeStubs]
from jaxtyping import Float

from viterbo.geometry.halfspaces import reference as halfspaces_reference

ConvexHull = _spatial.ConvexHull


def polytope_volume(
    B: Float[np.ndarray, " num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Trusted volume via SciPy's :class:`~scipy.spatial.ConvexHull`."""
    vertices = halfspaces_reference.enumerate_vertices(B, c, atol=atol)
    hull = ConvexHull(vertices, qhull_options="QJ")
    return float(hull.volume)
