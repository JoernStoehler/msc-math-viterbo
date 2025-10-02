"""Geometry helper tests for polytope constructions and transforms."""

from __future__ import annotations

import math

import numpy as np

from viterbo.geometry.halfspaces import enumerate_vertices
from viterbo.geometry.polytopes import (
    cartesian_product,
    cross_polytope,
    hypercube,
    mirror_polytope,
    random_affine_map,
    random_polytope,
    rotate_polytope,
    simplex_with_uniform_weights,
    translate_polytope,
)


def test_translate_polytope_updates_offsets() -> None:
    polytope = hypercube(2)
    translation = np.array([0.5, -0.25])
    translated = translate_polytope(polytope, translation)
    expected_c = polytope.c + polytope.B @ translation
    assert np.allclose(translated.c, expected_c)


def test_cartesian_product_dimensions_add() -> None:
    cube = hypercube(2)
    cross = cross_polytope(2)
    product = cartesian_product(cube, cross)
    assert product.dimension == cube.dimension + cross.dimension
    assert product.facets == cube.facets + cross.facets


def test_mirror_polytope_flips_coordinate() -> None:
    polytope = simplex_with_uniform_weights(3)
    mirrored = mirror_polytope(polytope, axes=(True, False, False))
    expected_B = polytope.B.copy()
    expected_B[:, 0] *= -1
    assert np.allclose(mirrored.B, expected_B)


def test_rotate_polytope_consistency_in_plane() -> None:
    polytope = hypercube(2)
    angle = math.pi / 4
    rotated = rotate_polytope(polytope, plane=(0, 1), angle=angle)
    rotation_matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    expected_B = polytope.B @ np.linalg.inv(rotation_matrix)
    assert np.allclose(rotated.B, expected_B)


def test_random_affine_map_is_deterministic_per_seed() -> None:
    rng_a = np.random.default_rng(10)
    rng_b = np.random.default_rng(10)
    matrix_a, translation_a = random_affine_map(4, rng=rng_a)
    matrix_b, translation_b = random_affine_map(4, rng=rng_b)
    assert np.allclose(matrix_a, matrix_b)
    assert np.allclose(translation_a, translation_b)
    assert np.linalg.cond(matrix_a) < 1e6


def test_random_polytope_facets_are_active() -> None:
    rng = np.random.default_rng(2024)
    polytope = random_polytope(3, rng=rng, name="random-3d-test")
    vertices = enumerate_vertices(polytope.B, polytope.c)
    for row, offset in zip(polytope.B, polytope.c, strict=False):
        assert np.any(np.isclose(vertices @ row, offset, atol=1e-9))
