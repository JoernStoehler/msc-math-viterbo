"""Tests for Euclidean volume helpers."""

from __future__ import annotations

import math

import numpy as np

from viterbo.geometry.polytopes import (
    Polytope,
    hypercube,
    random_polytope,
    simplex_with_uniform_weights,
)
from viterbo.geometry.volume import (
    hypercube_volume_inputs,
    polytope_volume_fast,
    polytope_volume_jax,
    polytope_volume_optimized,
    polytope_volume_reference,
)


def _volume_via_helpers(polytope: Polytope) -> tuple[float, float, float]:
    B, c = polytope.halfspace_data()
    reference = polytope_volume_reference(B, c)
    optimized = polytope_volume_fast(B, c)
    jax_result = polytope_volume_jax(B, c)
    return reference, optimized, jax_result


def test_hypercube_volume_matches_closed_form() -> None:
    cube = hypercube(4, radius=1.5)
    reference, fast, jax_result = _volume_via_helpers(cube)
    expected = (2 * 1.5) ** 4
    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(fast, expected, rel_tol=1e-9)
    assert math.isclose(jax_result, expected, rel_tol=1e-9)


def test_random_polytope_volumes_agree() -> None:
    rng = np.random.default_rng(42)
    polytope = random_polytope(4, rng=rng, name="random-test")
    reference, fast, jax_result = _volume_via_helpers(polytope)
    assert math.isclose(reference, fast, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(reference, jax_result, rel_tol=1e-9, abs_tol=1e-9)


def test_simplex_volume_positive() -> None:
    simplex = simplex_with_uniform_weights(4)
    reference, fast, jax_result = _volume_via_helpers(simplex)
    assert reference > 0
    assert math.isclose(reference, fast, rel_tol=1e-9)
    assert math.isclose(reference, jax_result, rel_tol=1e-9)


def test_hypercube_samples_match_expected_volume() -> None:
    matrix, offsets, expected = hypercube_volume_inputs(3, radius=2.0)
    reference = polytope_volume_reference(matrix, offsets)
    optimized = polytope_volume_optimized(matrix, offsets)
    jax_result = polytope_volume_jax(matrix, offsets)

    assert math.isclose(reference, expected, rel_tol=1e-9)
    assert math.isclose(reference, optimized, rel_tol=1e-9)
    assert math.isclose(reference, jax_result, rel_tol=1e-9)
