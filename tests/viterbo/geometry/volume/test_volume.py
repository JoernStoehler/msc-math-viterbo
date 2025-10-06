"""Tests for Euclidean volume helpers."""

from __future__ import annotations

import math

import jax
import pytest

from viterbo.geometry.polytopes import (
    Polytope,
    hypercube,
    random_polytope,
    simplex_with_uniform_weights,
)
from viterbo.geometry.volume import (
    hypercube_volume_inputs,
    polytope_volume_fast,
    polytope_volume_reference,
)


def _volumes(polytope: Polytope) -> tuple[float, float]:
    B, c = polytope.halfspace_data()
    return polytope_volume_reference(B, c), polytope_volume_fast(B, c)

@pytest.mark.goal_math
def test_hypercube_volume_matches_closed_form() -> None:
    """Both volume estimators match the analytic volume of a hypercube."""
    cube = hypercube(4, radius=1.5)
    reference, fast = _volumes(cube)
    expected = (2 * 1.5) ** 4
    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(fast, expected, rel_tol=1e-9)


@pytest.mark.goal_math
@pytest.mark.deep
def test_random_polytope_volumes_agree() -> None:
    """Reference and fast volume estimators agree on random 4D polytopes."""
    key = jax.random.PRNGKey(42)
    polytope = random_polytope(4, key=key, name="random-test")
    reference, fast = _volumes(polytope)
    assert math.isclose(reference, fast, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.goal_math
def test_simplex_volume_positive() -> None:
    """Volumes computed for the simplex are positive for both algorithms."""
    simplex = simplex_with_uniform_weights(4)
    reference, fast = _volumes(simplex)
    assert reference > 0
    assert fast > 0


@pytest.mark.goal_math
def test_hypercube_samples_match_expected_volume() -> None:
    """Generating hypercube volume inputs reproduces the analytic expectation."""
    matrix, offsets, expected = hypercube_volume_inputs(3, radius=2.0)
    reference = polytope_volume_reference(matrix, offsets)
    fast = polytope_volume_fast(matrix, offsets)
    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(fast, expected, rel_tol=1e-9)
