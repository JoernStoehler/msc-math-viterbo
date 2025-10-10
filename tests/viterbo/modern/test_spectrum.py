"""Ensure spectrum routines are still marked as unimplemented."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import spectrum
from viterbo.modern.types import PolytopeBundle


@pytest.mark.goal_code
@pytest.mark.smoke
def test_spectrum_stubs_raise_not_implemented() -> None:
    """Spectrum helpers should raise NotImplementedError for now."""

    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    normals = jnp.zeros((2, 3, 4))
    offsets = jnp.zeros((2, 3))

    with pytest.raises(NotImplementedError):
        spectrum.ehz_spectrum_reference(bundle, head=3)
    with pytest.raises(NotImplementedError):
        spectrum.ehz_spectrum_batched(normals, offsets, head=3)
