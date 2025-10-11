"""EHZ spectrum interfaces and expected output shapes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import atlas, spectrum


@pytest.mark.goal_code
@pytest.mark.smoke
@pytest.mark.xfail(reason="Spectrum algorithm not yet selected/implemented in modern.")
def test_ehz_spectrum_reference_returns_sequence_head() -> None:
    """Reference spectrum returns `head` leading actions in ascending order (pending algorithm)."""
    normals = [jnp.array([1.0, 0.0]), jnp.array([-1.0, 0.0]), jnp.array([0.0, 1.0]), jnp.array([0.0, -1.0])]
    offsets = [1.0, 1.0, 1.0, 1.0]
    vertices = [
        jnp.array([1.0, 1.0]),
        jnp.array([1.0, -1.0]),
        jnp.array([-1.0, 1.0]),
        jnp.array([-1.0, -1.0]),
    ]
    bundle = atlas.as_polytope(2, 4, 4, normals, offsets, vertices)
    head = 3
    seq = spectrum.ehz_spectrum_reference(bundle, head=head)
    assert len(seq) == head


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_spectrum_batched_returns_batch_by_head() -> None:
    """Batched spectrum returns array with shape (batch, head)."""
    normals = jnp.zeros((2, 3, 4), dtype=jnp.float64)
    offsets = jnp.zeros((2, 3), dtype=jnp.float64)
    head = 5
    arr = spectrum.ehz_spectrum_batched(normals, offsets, head=head)
    assert arr.shape == (2, head)
