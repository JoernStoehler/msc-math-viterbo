"""Tests for Euclidean volume helpers."""

from __future__ import annotations

import math

import numpy as np

from viterbo import (
    Polytope,
    hypercube,
    random_polytope,
    simplex_with_uniform_weights,
)
from viterbo.volume import polytope_volume_fast, polytope_volume_reference


def _volume_via_helpers(polytope: Polytope) -> tuple[float, float]:
    B, c = polytope.halfspace_data()
    return polytope_volume_reference(B, c), polytope_volume_fast(B, c)


def test_hypercube_volume_matches_closed_form() -> None:
    cube = hypercube(4, radius=1.5)
    reference, fast = _volume_via_helpers(cube)
    expected = (2 * 1.5) ** 4
    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(fast, expected, rel_tol=1e-9)


def test_random_polytope_volumes_agree() -> None:
    rng = np.random.default_rng(42)
    polytope = random_polytope(4, rng=rng, name="random-test")
    reference, fast = _volume_via_helpers(polytope)
    assert math.isclose(reference, fast, rel_tol=1e-9, abs_tol=1e-9)


def test_simplex_volume_positive() -> None:
    simplex = simplex_with_uniform_weights(4)
    reference, fast = _volume_via_helpers(simplex)
    assert reference > 0
    assert math.isclose(reference, fast, rel_tol=1e-9)
