"""Sample polytopes for volume estimators."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.halfspaces import samples as halfspace_samples


def hypercube_volume_inputs(
    dimension: int,
    *,
    radius: float = 1.0,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"], float]:
    """Return scaled hypercube ``(B, c, volume)`` triples for regression tests."""
    matrix, offsets = halfspace_samples.unit_hypercube_halfspaces(dimension)
    scaled_matrix = jnp.asarray(matrix)
    scaled_offsets = jnp.asarray(offsets) * float(radius)
    expected_volume = float((2 * radius) ** dimension)
    return scaled_matrix, scaled_offsets, expected_volume
