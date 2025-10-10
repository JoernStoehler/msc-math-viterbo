"""Check that EHZ capacity functions are still stubbed."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import capacity
from viterbo.modern.types import PolytopeBundle


@pytest.mark.goal_code
@pytest.mark.smoke
def test_capacity_stubs_raise_not_implemented() -> None:
    """EHZ capacity helpers should raise NotImplementedError for now."""

    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    normals = jnp.zeros((2, 3, 4))
    offsets = jnp.zeros((2, 3))

    with pytest.raises(NotImplementedError):
        capacity.ehz_capacity_reference(bundle)
    with pytest.raises(NotImplementedError):
        capacity.ehz_capacity_batched(normals, offsets, max_cycles=5)
