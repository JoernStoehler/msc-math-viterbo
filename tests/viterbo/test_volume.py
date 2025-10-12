"""Volume estimators: reference and padded interfaces (to be implemented)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.datasets import atlas
from viterbo.math import volume


@pytest.mark.goal_math
@pytest.mark.smoke
def test_volume_reference_matches_area_of_square() -> None:
    """Reference volume equals area 4.0 for square [-1,1]^2 (once implemented)."""
    normals = [
        jnp.array([1.0, 0.0]),
        jnp.array([-1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([0.0, -1.0]),
    ]
    offsets = [1.0, 1.0, 1.0, 1.0]
    vertices = [
        jnp.array([1.0, 1.0]),
        jnp.array([1.0, -1.0]),
        jnp.array([-1.0, 1.0]),
        jnp.array([-1.0, -1.0]),
    ]
    bundle = atlas.as_polytope(2, 4, 4, normals, offsets, vertices)
    vol = volume.volume_reference(bundle.vertices)
    assert jnp.isclose(vol, 4.0, rtol=1e-12, atol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_volume_padded_accepts_batched_halfspaces_signature() -> None:
    """Batched volume interface accepts (normals, offsets) and a method flag."""
    normals = jnp.zeros((2, 3, 4), dtype=jnp.float64)
    offsets = jnp.zeros((2, 3), dtype=jnp.float64)
    vol = volume.volume_padded(normals, offsets, method="monte_carlo")
    assert vol.shape == (2,)
