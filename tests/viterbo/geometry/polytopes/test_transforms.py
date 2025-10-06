"""Geometry helper tests for polytope constructions and transforms."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.goal_code
def test_translate_polytope_updates_offsets() -> None:
    """Translating a polytope shifts its half-space offsets by B @ translation."""
    polytope = hypercube(2)
    translation = jnp.array([0.5, -0.25])
    translated = translate_polytope(polytope, translation)
    expected_c = polytope.c + polytope.B @ translation
    np.testing.assert_allclose(
        np.asarray(translated.c), np.asarray(expected_c), rtol=1e-9, atol=0.0
    )

@pytest.mark.goal_math
def test_cartesian_product_dimensions_add() -> None:
    """Cartesian products add both dimension and facet counts."""
    cube = hypercube(2)
    cross = cross_polytope(2)
    product = cartesian_product(cube, cross)
    assert product.dimension == cube.dimension + cross.dimension
    assert product.facets == cube.facets + cross.facets


@pytest.mark.goal_code
def test_mirror_polytope_flips_coordinate() -> None:
    """Mirroring across an axis negates the corresponding facet coefficients."""
    polytope = simplex_with_uniform_weights(3)
    mirrored = mirror_polytope(polytope, axes=(True, False, False))
    expected_B = polytope.B.copy()
    expected_B = expected_B.at[:, 0].multiply(-1)
    np.testing.assert_allclose(np.asarray(mirrored.B), np.asarray(expected_B), rtol=1e-9, atol=0.0)

@pytest.mark.goal_math
def test_rotate_polytope_consistency_in_plane() -> None:
    """Rotation updates facet normals according to the inverse rotation matrix."""
    polytope = hypercube(2)
    angle = math.pi / 4
    rotated = rotate_polytope(polytope, plane=(0, 1), angle=angle)
    rotation_matrix = jnp.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    expected_B = polytope.B @ jnp.linalg.inv(rotation_matrix)
    np.testing.assert_allclose(np.asarray(rotated.B), np.asarray(expected_B), rtol=1e-9, atol=0.0)

@pytest.mark.goal_code
def test_random_affine_map_is_deterministic_per_seed() -> None:
    """Random affine maps are reproducible with a fixed seed and well-conditioned."""
    key = jax.random.PRNGKey(10)
    matrix_a, translation_a = random_affine_map(4, key=key)
    matrix_b, translation_b = random_affine_map(4, key=key)
    np.testing.assert_allclose(np.asarray(matrix_a), np.asarray(matrix_b), rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(translation_a), np.asarray(translation_b), rtol=1e-9, atol=0.0
    )

    split_a, split_b = jax.random.split(key)
    matrix_c, translation_c = random_affine_map(4, key=split_a)
    matrix_d, translation_d = random_affine_map(4, key=split_b)
    assert not np.array_equal(np.asarray(matrix_c), np.asarray(matrix_d))
    assert not np.array_equal(np.asarray(translation_c), np.asarray(translation_d))
    assert np.linalg.cond(np.asarray(matrix_a)) < 1e6


@pytest.mark.goal_math
@pytest.mark.deep
def test_random_polytope_facets_are_active() -> None:
    """Every generated facet is tight for at least one enumerated vertex."""
    key = jax.random.PRNGKey(2024)
    polytope = random_polytope(3, key=key, name="random-3d-test")
    vertices = enumerate_vertices(polytope.B, polytope.c)
    # Each facet touches at least one vertex at equality
    rows = np.asarray(polytope.B, dtype=float)
    offs = np.asarray(polytope.c, dtype=float)
    for row, offset in zip(rows, offs, strict=False):
        assert np.any(np.isclose(np.asarray(vertices, dtype=float) @ row, offset, atol=1e-9))  # type: ignore[reportUnknownArgumentType]
