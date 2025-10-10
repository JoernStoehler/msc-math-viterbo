"""Ensure cycle routines remain stubbed."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import cycles
from viterbo.modern.types import PolytopeBundle


@pytest.mark.goal_code
@pytest.mark.smoke
def test_cycle_stubs_raise_not_implemented() -> None:
    """Cycle helpers should raise NotImplementedError for now."""

    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    normals = jnp.zeros((2, 3, 4))
    offsets = jnp.zeros((2, 3))
    padded_vertices = jnp.zeros((2, 5, 4))

    with pytest.raises(NotImplementedError):
        cycles.minimum_cycle_reference(bundle)
    with pytest.raises(NotImplementedError):
        cycles.minimum_cycle_batched(normals, offsets, padded_vertices=padded_vertices)
