"""Friendly entry points for smoke testing the tooling."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from viterbo.symplectic.core import normalize_vector

_DIMENSION_AXIS = "dimension"


def hello_numpy(
    name: str,
    sample: Float[np.ndarray, _DIMENSION_AXIS],
    *,
    greeting: str = "Hello",
) -> str:
    """Return a short greeting that includes a normalized NumPy sample.

    Args:
      name: Person or system to greet.
      sample: One-dimensional NumPy array normalized for display. Shape ``(dimension,)``.
      greeting: Leading salutation to use in the greeting.

    Returns:
      Human-readable message embedding the normalized sample.
    """
    normalized_sample: Float[np.ndarray, _DIMENSION_AXIS] = normalize_vector(sample)
    components = ", ".join(f"{value:.3f}" for value in normalized_sample)
    message = f"{greeting}, {name}! Unit sample: [{components}]"
    return message
