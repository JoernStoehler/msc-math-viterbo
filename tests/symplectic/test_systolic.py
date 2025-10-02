"""Tests for the systolic ratio helper."""

from __future__ import annotations

import numpy as np

from viterbo.geometry.polytopes import simplex_with_uniform_weights, translate_polytope
from viterbo.symplectic.systolic import systolic_ratio


def test_systolic_ratio_translation_invariant() -> None:
    simplex = simplex_with_uniform_weights(4)
    translated = translate_polytope(simplex, np.array([0.2, -0.3, 0.4, 0.1]))
    assert np.isclose(systolic_ratio(simplex), systolic_ratio(translated), rtol=1e-9)


def test_systolic_ratio_scale_invariant() -> None:
    simplex = simplex_with_uniform_weights(4)
    B, c = simplex.halfspace_data()
    scaled_ratio = systolic_ratio(B, 1.5 * c)
    assert np.isclose(systolic_ratio(simplex), scaled_ratio, rtol=1e-9)


def test_simplex_ratio_positive() -> None:
    simplex = simplex_with_uniform_weights(4)
    assert systolic_ratio(simplex) > 0.0
