"""EHZ capacity semantics for reference and per-instance APIs."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import atlas, capacity
from viterbo.types import Polytope


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
    c = capacity.ehz_capacity_reference(bundle.normals, bundle.offsets, bundle.vertices)
    assert jnp.isfinite(c)
    assert c == pytest.approx(4.0, rel=1e-12, abs=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_capacity_reference_manual_batching_pattern() -> None:
    """Manual loops over bundles reproduce batching behaviour without a helper."""

    def _square_bundle(radius: float) -> Polytope:
        normals = [
            jnp.array([1.0, 0.0]),
            jnp.array([-1.0, 0.0]),
            jnp.array([0.0, 1.0]),
            jnp.array([0.0, -1.0]),
        ]
        offsets = [radius] * 4
        vertices = [
            jnp.array([radius, radius]),
            jnp.array([radius, -radius]),
            jnp.array([-radius, radius]),
            jnp.array([-radius, -radius]),
        ]
        return atlas.as_polytope(2, 4, 4, normals, offsets, vertices)

    bundles = [_square_bundle(1.0), _square_bundle(2.0)]
    capacities = []
    for bundle in bundles:
        capacities.append(
            capacity.ehz_capacity_reference(bundle.normals, bundle.offsets, bundle.vertices)
        )

    assert capacities[0] == pytest.approx(4.0, rel=1e-12, abs=0.0)
    assert capacities[1] == pytest.approx(16.0, rel=1e-12, abs=0.0)
    assert not hasattr(capacity, "ehz_capacity_batched")
