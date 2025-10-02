"""Friendly entry points for smoke testing the tooling."""

from __future__ import annotations

from viterbo.symplectic.core import Vector, normalize_vector


def hello_numpy(name: str, sample: Vector, *, greeting: str = "Hello") -> str:
    """
    Return a short greeting that includes a normalized NumPy sample.

    Parameters
    ----------
    name:
        Person or system to greet.
    sample:
        One-dimensional NumPy array that will be normalized for display.
    greeting:
        Leading salutation to use in the greeting.

    Returns
    -------
    str
        Human-readable message embedding the normalized sample.

    """
    normalized_sample: Vector = normalize_vector(sample)  # shape: (n,)
    components = ", ".join(f"{value:.3f}" for value in normalized_sample)
    message = f"{greeting}, {name}! Unit sample: [{components}]"
    return message
