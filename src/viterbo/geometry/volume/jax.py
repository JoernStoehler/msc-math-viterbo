"""JAX-compatible volume estimators."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Protocol, cast

import numpy as np
from jaxtyping import Float

from viterbo.geometry.halfspaces import jax as halfspaces_jax
from viterbo.geometry.volume import _shared


class _ConvexHullResult(Protocol):
    volume: float


class _DelaunayResult(Protocol):
    simplices: np.ndarray


class _ConvexHullFactory(Protocol):
    def __call__(
        self,
        points: np.ndarray,
        *,
        qhull_options: str | None = ...,  # noqa: D401 - see implementation docs
    ) -> _ConvexHullResult:
        """Return the convex hull of ``points``."""
        ...


class _DelaunayFactory(Protocol):
    def __call__(
        self,
        points: np.ndarray,
        *,
        qhull_options: str | None = ...,  # noqa: D401 - see implementation docs
    ) -> _DelaunayResult:
        """Return the Delaunay triangulation of ``points``."""
        ...


@lru_cache(1)
def _load_spatial() -> tuple[_ConvexHullFactory, _DelaunayFactory, type[Exception]]:
    """Return SciPy spatial factories with static typing."""

    spatial = importlib.import_module("scipy.spatial")
    convex_hull = cast(_ConvexHullFactory, getattr(spatial, "ConvexHull"))
    delaunay = cast(_DelaunayFactory, getattr(spatial, "Delaunay"))
    qhull_error = cast(type[Exception], getattr(spatial, "QhullError"))
    return convex_hull, delaunay, qhull_error


def polytope_volume(
    B: Float[np.ndarray, "num_facets dimension"],
    c: Float[np.ndarray, "num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    """Estimate volume using JAX-powered vertex solves with SciPy triangulation."""
    vertices = halfspaces_jax.enumerate_vertices(B, c, atol=atol)
    vertices_np = np.asarray(vertices, dtype=float)

    if vertices_np.size == 0:
        msg = "No vertices found; polytope may be empty or unbounded."
        raise ValueError(msg)

    convex_hull, delaunay, qhull_error = _load_spatial()

    try:
        triangulation = delaunay(vertices_np, qhull_options="QJ")
    except qhull_error:
        hull = convex_hull(vertices_np, qhull_options="QJ")
        return float(hull.volume)

    simplices = vertices_np[triangulation.simplices]
    return _shared.volume_of_simplices(simplices)
