"""Sample polytopes for volume estimators."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from viterbo.geometry.halfspaces import samples as halfspace_samples


def hypercube_volume_inputs(
    dimension: int,
    *,
    radius: float = 1.0,
) -> tuple[Float[np.ndarray, " num_facets dimension"], Float[np.ndarray, " num_facets"], float]:
    """Return scaled hypercube ``(B, c, volume)`` triples for regression tests."""
    matrix, offsets = halfspace_samples.unit_hypercube_halfspaces(dimension)
    scaled_matrix = matrix.copy()
    scaled_offsets = offsets * radius
    expected_volume = float((2 * radius) ** dimension)
    return scaled_matrix, scaled_offsets, expected_volume
