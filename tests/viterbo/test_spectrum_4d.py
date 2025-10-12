"""Spectrum smoke tests for the 4D modern implementation."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from viterbo.math import spectrum


@pytest.mark.goal_math
@pytest.mark.smoke
def test_ehz_spectrum_reference_returns_head_actions_4d() -> None:
    """4D spectrum returns up to ``head`` finite, ascending actions."""
    normals = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    offsets = jnp.ones((8,), dtype=jnp.float64)
    head = 4
    seq = spectrum.ehz_spectrum_reference(normals, offsets, head=head)
    assert len(seq) <= head
    assert all(math.isfinite(x) and x >= 0.0 for x in seq)
    assert seq == sorted(seq)
