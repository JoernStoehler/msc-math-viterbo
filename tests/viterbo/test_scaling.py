"""Homogeneity and scale-invariance properties (capacity, volume, systolic)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from viterbo._wrapped import spatial as _spatial
from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals
from viterbo.math.systolic import systolic_ratio
from viterbo.math.volume import polytope_volume_reference


def _random_hull_4d(seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    V = jax.random.normal(key, (5, 4), dtype=jnp.float64)
    try:
        eq = _spatial.convex_hull_equations(V)
    except _spatial.QhullError:
        assume(False)
        raise AssertionError
    B = jnp.asarray(eq[:, :-1], dtype=jnp.float64)
    c = jnp.asarray(-eq[:, -1], dtype=jnp.float64)
    return B, c


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=3, deadline=None)
@given(st.integers(min_value=0, max_value=2**31 - 1), st.floats(min_value=0.5, max_value=2.0))
def test_capacity_scales_quadratically_in_4d(seed: int, scale: float) -> None:
    """c_EHZ(aK) = a^2 c_EHZ(K) for convex polytopes in R^4."""
    B, c = _random_hull_4d(seed)
    base = ehz_capacity_reference_facet_normals(B, c)
    scaled = ehz_capacity_reference_facet_normals(B, jnp.asarray(scale, dtype=jnp.float64) * c)
    assert scaled == pytest.approx((scale**2) * base, rel=1e-9, abs=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=3, deadline=None)
@given(st.integers(min_value=0, max_value=2**31 - 1), st.floats(min_value=0.5, max_value=2.0))
def test_volume_scales_as_degree_four_in_4d(seed: int, scale: float) -> None:
    """vol(aK) = a^4 vol(K) for polytopes in R^4."""
    B, c = _random_hull_4d(seed)
    base = polytope_volume_reference(B, c)
    scaled = polytope_volume_reference(B, jnp.asarray(scale, dtype=jnp.float64) * c)
    assert scaled == pytest.approx((scale**4) * base, rel=1e-9, abs=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
@settings(max_examples=3, deadline=None)
@given(st.integers(min_value=0, max_value=2**31 - 1), st.floats(min_value=0.5, max_value=2.0))
def test_systolic_ratio_is_scale_invariant(seed: int, scale: float) -> None:
    """sys(aK) = sys(K) since c scales quadratically and volume as degree 4."""
    B, c = _random_hull_4d(seed)
    base = systolic_ratio(B, c)
    scaled = systolic_ratio(B, jnp.asarray(scale, dtype=jnp.float64) * c)
    assert math.isfinite(base)
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-12)
