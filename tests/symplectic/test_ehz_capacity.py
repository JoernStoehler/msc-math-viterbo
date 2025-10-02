"""Tests for the combinatorial EHZ capacity implementation."""

from __future__ import annotations

import numpy as np
import pytest

from viterbo.geometry.polytopes import (
    Polytope,
    catalog,
    simplex_with_uniform_weights,
    truncated_simplex_four_dim,
)
from viterbo.symplectic.capacity import compute_ehz_capacity
from viterbo.symplectic.capacity_fast import compute_ehz_capacity_fast


def _simplex() -> Polytope:
    return simplex_with_uniform_weights(4, name="simplex-test")


@pytest.mark.parametrize(
    "polytope",
    [poly for poly in catalog() if poly.reference_capacity is not None],
    ids=lambda poly: poly.name,
)
def test_capacity_matches_reference_value(polytope: Polytope) -> None:
    """Reference polytopes reproduce their known EHZ capacities."""

    B, c = polytope.halfspace_data()
    capacity = compute_ehz_capacity(B, c)
    assert polytope.reference_capacity is not None
    assert np.isclose(capacity, polytope.reference_capacity, atol=1e-9)


def test_capacity_scales_quadratically_under_dilation() -> None:
    r"""Scaling the polytope dilates the capacity by the square factor."""

    polytope = _simplex()
    B, c = polytope.halfspace_data()
    base_capacity = compute_ehz_capacity(B, c)

    scale = 1.5
    scaled_capacity = compute_ehz_capacity(B, scale * c)

    assert np.isclose(scaled_capacity, (scale**2) * base_capacity, atol=1e-8)


def test_capacity_is_translation_invariant() -> None:
    """Rigid translations of the polytope leave ``c_EHZ`` unchanged."""

    polytope = _simplex()
    B, c = polytope.halfspace_data()
    base_capacity = compute_ehz_capacity(B, c)

    translation = np.array([0.3, -0.2, 0.1, -0.4])
    translated_c = c + B @ translation

    translated_capacity = compute_ehz_capacity(B, translated_c)

    assert np.isclose(translated_capacity, base_capacity, atol=1e-9)


def test_truncated_simplex_matches_known_subset_action() -> None:
    """Adding an extra facet leaves the optimal action unchanged."""

    polytope = truncated_simplex_four_dim()
    B, c = polytope.halfspace_data()
    capacity = compute_ehz_capacity(B, c)

    assert polytope.reference_capacity is not None
    assert np.isclose(capacity, polytope.reference_capacity, atol=1e-9)


def test_two_dimensional_simplex_matches_fast_capacity() -> None:
    """The 2D simplex yields a finite, consistent capacity across implementations."""

    polytope = simplex_with_uniform_weights(2, name="simplex-2d-test")
    B, c = polytope.halfspace_data()

    reference = compute_ehz_capacity(B, c)
    optimized = compute_ehz_capacity_fast(B, c)

    assert np.isfinite(reference)
    assert np.isfinite(optimized)
    assert np.isclose(reference, optimized, atol=1e-9)
