"""Check that volume routines remain stubbed out."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import volume
from viterbo.modern.types import PolytopeBundle


@pytest.mark.goal_code
@pytest.mark.smoke
def test_volume_stubs_raise_not_implemented() -> None:
    """Volume helpers should raise NotImplementedError until implemented."""

    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    normals = jnp.zeros((2, 3, 4))
    offsets = jnp.zeros((2, 3))

    with pytest.raises(NotImplementedError):
        volume.volume_reference(bundle)
    with pytest.raises(NotImplementedError):
        volume.volume_padded(normals, offsets, method="monte_carlo")
