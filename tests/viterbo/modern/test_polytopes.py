"""Ensure polytope construction stubs signal incomplete implementations."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo.modern import polytopes
from viterbo.modern.types import PolytopeBundle


@pytest.mark.goal_code
@pytest.mark.smoke
def test_polytope_stubs_raise_not_implemented() -> None:
    """Each public constructor should raise NotImplementedError until filled in."""

    normals = jnp.zeros((2, 3))
    offsets = jnp.zeros((2,))
    vertices = jnp.zeros((4, 3))
    bundle = PolytopeBundle(halfspaces=None, vertices=None)

    with pytest.raises(NotImplementedError):
        polytopes.build_from_halfspaces(normals, offsets)
    with pytest.raises(NotImplementedError):
        polytopes.build_from_vertices(vertices)
    with pytest.raises(NotImplementedError):
        polytopes.complete_incidence(bundle)
    with pytest.raises(NotImplementedError):
        polytopes.pad_polytope_bundle(bundle, target_facets=3, target_vertices=5)
