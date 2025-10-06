"""Tests for the systolic ratio helper."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from viterbo.geometry.polytopes import simplex_with_uniform_weights, translate_polytope
from viterbo.symplectic.systolic import systolic_ratio


@pytest.mark.goal_math
def test_systolic_ratio_translation_invariant() -> None:
    """Translating a simplex leaves the systolic ratio unchanged."""
    simplex = simplex_with_uniform_weights(4)
    translated = translate_polytope(simplex, jnp.array([0.2, -0.3, 0.4, 0.1]))
    assert math.isclose(systolic_ratio(simplex), systolic_ratio(translated), rel_tol=1e-9)


@pytest.mark.goal_math
def test_systolic_ratio_scale_invariant() -> None:
    """Scaling a polytope rescales its halfspaces without changing the ratio."""
    simplex = simplex_with_uniform_weights(4)
    B, c = simplex.halfspace_data()
    scaled_ratio = systolic_ratio(B, 1.5 * c)
    assert math.isclose(systolic_ratio(simplex), scaled_ratio, rel_tol=1e-9)


@pytest.mark.goal_math
def test_simplex_ratio_positive() -> None:
    """The systolic ratio of the canonical simplex is strictly positive."""
    simplex = simplex_with_uniform_weights(4)
    assert systolic_ratio(simplex) > 0.0


@pytest.mark.goal_code
def test_raw_halfspace_input_validates_shapes() -> None:
    """Shape mismatches in halfspace inputs trigger validation errors."""
    B = np.array([1.0, -1.0])
    c = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        systolic_ratio(B, c)  # type: ignore[reportArgumentType]

    B_matrix = np.eye(4)
    c_matrix = np.ones((4, 1))
    with pytest.raises(ValueError):
        systolic_ratio(B_matrix, c_matrix)  # type: ignore[reportArgumentType]

    with pytest.raises(ValueError):
        systolic_ratio(B_matrix, np.ones(3))  # type: ignore[reportArgumentType]
