"""EHZ capacity semantics for reference and batched APIs."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import atlas, capacity, polytopes
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
    c = capacity.ehz_capacity_reference(bundle)
    assert jnp.isfinite(c)
    assert c == pytest.approx(4.0, rel=1e-12, abs=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_ehz_capacity_batched_signature_and_shapes() -> None:
    """Batched capacity returns per-sample scalars with NaN padding."""
    vertices = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    valid = polytopes.build_from_vertices(vertices)
    invalid = Polytope(
        normals=jnp.zeros((0, 4), dtype=jnp.float64),
        offsets=jnp.zeros((0,), dtype=jnp.float64),
        vertices=jnp.zeros((0, 4), dtype=jnp.float64),
        incidence=jnp.zeros((0, 0), dtype=bool),
    )
    caps = capacity.ehz_capacity_batched([valid, invalid], solver="facet-normal-reference")
    assert caps.shape == (2,)
    assert jnp.isfinite(caps[0])
    assert jnp.isnan(caps[1])
