"""Volume estimators: reference and padded interfaces (to be implemented)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.datasets2 import generators
from viterbo.math import volume


@pytest.mark.goal_math
@pytest.mark.smoke
def test_volume_reference_matches_area_of_square() -> None:
    """Reference volume equals area 4.0 for square [-1,1]^2."""
    vertices = jnp.asarray(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    vol = volume.volume_reference(vertices)
    assert jnp.isclose(vol, 4.0, rtol=1e-12, atol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_volume_padded_accepts_batched_halfspaces_signature() -> None:
    """Batched volume interface accepts (normals, offsets) and a method flag."""
    normals = jnp.zeros((2, 3, 4), dtype=jnp.float64)
    offsets = jnp.zeros((2, 3), dtype=jnp.float64)
    vol = volume.volume_padded(normals, offsets, method="monte_carlo")
    assert vol.shape == (2,)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_volume_hypercube_4d_closed_form() -> None:
    """Hypercube [-1,1]^4 volume equals 16."""
    cube = generators.hypercube(4, radius=1.0)
    v = volume.volume_reference(cube.vertices)
    assert jnp.isclose(v, 16.0, rtol=1e-12, atol=2e-9)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_volume_simplex_4d_closed_form() -> None:
    """Right 4D simplex conv(0, e_i) has volume 1/24."""
    simp = generators.simplex(4)
    v = volume.volume_reference(simp.vertices)
    assert jnp.isclose(v, 1.0 / 24.0, rtol=1e-12, atol=2e-11)
