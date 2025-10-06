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


def test_hypercube_volume_matches_closed_form() -> None:
    cube = hypercube(4, radius=1.5)
    reference, fast = _volumes(cube)
    expected = (2 * 1.5) ** 4
    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(fast, expected, rel_tol=1e-9)


@pytest.mark.deep
def test_random_polytope_volumes_agree() -> None:
    key = jax.random.PRNGKey(42)
    polytope = random_polytope(4, key=key, name="random-test")
    reference, fast = _volumes(polytope)
    assert math.isclose(reference, fast, rel_tol=1e-9, abs_tol=1e-9)


def test_simplex_volume_positive() -> None:
    simplex = simplex_with_uniform_weights(4)
    reference, fast = _volumes(simplex)
    assert reference > 0
    assert fast > 0


def test_hypercube_samples_match_expected_volume() -> None:
    matrix, offsets, expected = hypercube_volume_inputs(3, radius=2.0)
    reference = polytope_volume_reference(matrix, offsets)
    fast = polytope_volume_fast(matrix, offsets)
    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(fast, expected, rel_tol=1e-9)
