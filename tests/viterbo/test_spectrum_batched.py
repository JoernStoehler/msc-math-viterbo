"""Batched spectrum semantics for the modern API (smoke)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import spectrum


@pytest.mark.goal_code
@pytest.mark.smoke
def test_spectrum_batched_shape_and_nan_padding() -> None:
    """Batched spectrum returns (batch, head) with NaN padding for missing entries."""
    # Batch element 0: valid 4D cube
    normals4 = jnp.array(
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
    offsets4 = jnp.ones((8,), dtype=jnp.float64)
    # Batch element 1: 2D system (invalid for 4D spectrum) should return NaNs
    normals2 = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    offsets2 = jnp.ones((4,), dtype=jnp.float64)

    # Broadcast 2D normals/offsets to 4D by embedding with trailing zeros so shapes match
    normals2_4d = jnp.hstack([normals2, jnp.zeros((normals2.shape[0], 2), dtype=jnp.float64)])
    # Pad to 8 facets with degenerate constraints to exercise NaN padding semantics
    pad_n = 8 - normals2_4d.shape[0]
    normals2_4d = jnp.vstack([normals2_4d, jnp.zeros((pad_n, 4), dtype=jnp.float64)])
    offsets2_4d = jnp.concatenate([offsets2, -jnp.ones((pad_n,), dtype=jnp.float64)])
    normals = jnp.stack([normals4, normals2_4d])
    offsets = jnp.stack([offsets4, offsets2_4d])
    head = 6
    arr = spectrum.ehz_spectrum_batched(normals, offsets, head=head)
    assert arr.shape == (2, head)
    # Row 0: expect some finite prefix and NaN padding or fully finite short head
    finite0 = jnp.isfinite(arr[0])
    assert jnp.any(finite0)
    # Row 1: invalid dimension → all NaNs
    assert jnp.all(jnp.isnan(arr[1]))
