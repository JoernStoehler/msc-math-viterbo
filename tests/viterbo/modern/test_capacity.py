"""EHZ capacity semantics for reference and batched APIs."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import atlas, capacity


@pytest.mark.goal_math
@pytest.mark.smoke
def test_ehz_capacity_reference_for_square_nonnegative_scalar() -> None:
    """Reference EHZ capacity returns a finite, nonnegative scalar (exact value TBD)."""
    normals = [jnp.array([1.0, 0.0]), jnp.array([-1.0, 0.0]), jnp.array([0.0, 1.0]), jnp.array([0.0, -1.0])]
    offsets = [1.0, 1.0, 1.0, 1.0]
    vertices = [
        jnp.array([1.0, 1.0]),
        jnp.array([1.0, -1.0]),
        jnp.array([-1.0, 1.0]),
        jnp.array([-1.0, -1.0]),
    ]
    bundle = atlas.as_polytope(2, 4, 4, normals, offsets, vertices)
    c = capacity.ehz_capacity_reference(bundle)
    assert jnp.isfinite(c)
    assert c >= 0.0


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_capacity_batched_signature_and_shapes() -> None:
    """Batched capacity returns per-sample scalars with shape (batch,)."""
    normals = jnp.zeros((2, 3, 4), dtype=jnp.float64)
    offsets = jnp.zeros((2, 3), dtype=jnp.float64)
    caps = capacity.ehz_capacity_batched(normals, offsets, max_cycles=5)
    assert caps.shape == (2,)
