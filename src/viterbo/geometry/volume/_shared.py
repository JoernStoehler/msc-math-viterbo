"""Shared kernels for volume estimators."""

from __future__ import annotations

import math

import numpy as np
from jaxtyping import Float


def volume_of_simplices(
    simplex_vertices: Float[np.ndarray, "num_simplices vertices dimension"],
) -> float:
    """Return the total volume of stacked simplices."""
    base = simplex_vertices[:, 0, :]
    edges = simplex_vertices[:, 1:, :] - base[:, None, :]
    determinants = np.linalg.det(edges)
    dimension = simplex_vertices.shape[2]
    return float(np.sum(np.abs(determinants)) / math.factorial(int(dimension)))
