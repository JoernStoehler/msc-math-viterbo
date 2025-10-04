"""Curated half-space datasets for regression tests and examples."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float


def unit_hypercube_halfspaces(
    dimension: int,
) -> tuple[Float[np.ndarray, " num_facets dimension"], Float[np.ndarray, " num_facets"]]:
    """Return ``Bx ≤ c`` for the axis-aligned unit hypercube centered at the origin."""
    if dimension <= 0:
        msg = "dimension must be positive"
        raise ValueError(msg)

    identity = np.eye(dimension, dtype=float)
    matrix = np.vstack((identity, -identity))
    offsets = np.ones(2 * dimension, dtype=float)
    return matrix, offsets


def unit_square_halfspaces() -> tuple[
    Float[np.ndarray, " num_facets dimension"],
    Float[np.ndarray, " num_facets"],
]:
    """Return a cached 2D unit square description for convenience."""
    return unit_hypercube_halfspaces(2)
