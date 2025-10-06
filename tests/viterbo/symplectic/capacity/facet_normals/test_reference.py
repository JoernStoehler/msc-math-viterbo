"""Tests for the combinatorial EHZ capacity implementation (reference)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from tests._utils.baselines import load_baseline
from viterbo.geometry.polytopes import (
    Polytope,
    catalog,
    simplex_with_uniform_weights,
    truncated_simplex_four_dim,
)
from viterbo.symplectic.capacity import compute_ehz_capacity_reference
from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast


def _simplex() -> Polytope:
    return simplex_with_uniform_weights(4, name="simplex-test")

_BASELINE_CAPACITIES = {
    entry["name"]: entry["reference_capacity"]
    for entry in load_baseline("ehz_capacity_reference")
}
_POLYTOPE_LOOKUP = {poly.name: poly for poly in catalog()}

@pytest.mark.parametrize(
    "polytope_name",
    sorted(_BASELINE_CAPACITIES.keys()),
)
def test_capacity_matches_baseline(polytope_name: str) -> None:
    """Reference polytopes reproduce their known EHZ capacities."""

    expected = _BASELINE_CAPACITIES[polytope_name]
    polytope = _POLYTOPE_LOOKUP[polytope_name]
    B, c = polytope.halfspace_data()
    capacity = compute_ehz_capacity_reference(B, c)
    assert math.isclose(capacity, expected, rel_tol=0.0, abs_tol=1e-9)


def test_capacity_scales_quadratically_under_dilation() -> None:
    r"""Scaling the polytope dilates the capacity by the square factor."""

    polytope = _simplex()
    B, c = polytope.halfspace_data()
    base_capacity = compute_ehz_capacity_reference(B, c)

    scale = 1.5
    scaled_capacity = compute_ehz_capacity_reference(B, scale * c)

    assert math.isclose(scaled_capacity, (scale**2) * base_capacity, rel_tol=0.0, abs_tol=1e-8)


def test_capacity_is_translation_invariant() -> None:
    """Rigid translations of the polytope leave ``c_EHZ`` unchanged."""

    polytope = _simplex()
    B, c = polytope.halfspace_data()
    base_capacity = compute_ehz_capacity_reference(B, c)

    translation = jnp.array([0.3, -0.2, 0.1, -0.4])
    translated_c = c + B @ translation

    translated_capacity = compute_ehz_capacity_reference(B, translated_c)

    assert math.isclose(translated_capacity, base_capacity, rel_tol=0.0, abs_tol=1e-9)


def test_truncated_simplex_matches_known_subset_action() -> None:
    """Adding an extra facet leaves the optimal action unchanged."""

    polytope = truncated_simplex_four_dim()
    B, c = polytope.halfspace_data()
    capacity = compute_ehz_capacity_reference(B, c)

    assert polytope.reference_capacity is not None
    assert math.isclose(capacity, polytope.reference_capacity, rel_tol=0.0, abs_tol=1e-9)


def test_two_dimensional_simplex_matches_fast_capacity() -> None:
    """The 2D simplex yields a finite, consistent capacity across implementations."""

    polytope = simplex_with_uniform_weights(2, name="simplex-2d-test")
    B, c = polytope.halfspace_data()

    reference = compute_ehz_capacity_reference(B, c)
    optimized = compute_ehz_capacity_fast(B, c)

    assert math.isfinite(reference)
    assert math.isfinite(optimized)
    assert math.isclose(reference, optimized, rel_tol=0.0, abs_tol=1e-9)
