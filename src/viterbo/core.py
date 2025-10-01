"""Numerical building blocks for experimenting with the Viterbo conjecture."""

from __future__ import annotations

from typing import Final

import numpy as np
from jaxtyping import Float

Vector = Float[np.ndarray, " n"]
ZERO_TOLERANCE: Final[float] = 1e-12


def normalize_vector(vector: Vector) -> Vector:
    """
    Return a unit vector pointing in the same direction as ``vector``.

    Parameters
    ----------
    vector:
        One-dimensional NumPy array representing the vector to normalize.

    Returns
    -------
    numpy.ndarray
        A unit vector pointing in the same direction as ``vector``.

    Raises
    ------
    ValueError
        If ``vector`` is (numerically) the zero vector.

    Examples
    --------
    >>> import numpy as np
    >>> from viterbo import normalize_vector
    >>> normalize_vector(np.array([3.0, 4.0]))
    array([0.6, 0.8])

    """
    norm: float = float(np.linalg.norm(vector))
    if norm <= ZERO_TOLERANCE:
        raise ValueError("Cannot normalize a vector with near-zero magnitude.")

    normalized: Vector = vector / norm  # shape: (n,)
    return normalized
