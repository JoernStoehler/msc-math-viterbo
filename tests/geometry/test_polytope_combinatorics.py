"""Tests for cached combinatorial data of polytopes."""

from __future__ import annotations

import numpy as np

from viterbo.geometry.polytopes import (
    PolytopeCombinatorics,
    halfspaces_from_vertices,
    hypercube,
    polytope_combinatorics,
    polytope_combinatorics_jax,
    polytope_combinatorics_optimized,
    polytope_fingerprint,
    vertices_from_halfspaces,
    vertices_from_halfspaces_jax,
    vertices_from_halfspaces_optimized,
)


def _sort_halfspaces(
    B: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    key = np.lexsort(np.column_stack((B, c)).T)
    return B[key], c[key]


def test_polytope_combinatorics_square_facets() -> None:
    square = hypercube(2)
    combinatorics = polytope_combinatorics(square, use_cache=False)

    assert isinstance(combinatorics, PolytopeCombinatorics)
    assert combinatorics.vertices.shape == (4, 2)
    assert combinatorics.facet_adjacency.shape == (4, 4)
    # Each facet of a square touches exactly two neighbours.
    degree = combinatorics.facet_adjacency.sum(axis=1)
    assert np.all(degree == 2)


def test_polytope_combinatorics_variants_match() -> None:
    cube = hypercube(3)
    baseline = polytope_combinatorics(cube, use_cache=False)
    optimized = polytope_combinatorics_optimized(cube, use_cache=False)
    jax_variant = polytope_combinatorics_jax(cube, use_cache=False)

    assert np.array_equal(optimized.facet_adjacency, baseline.facet_adjacency)
    assert np.array_equal(jax_variant.facet_adjacency, baseline.facet_adjacency)

    assert np.allclose(optimized.vertices, baseline.vertices)
    assert np.allclose(jax_variant.vertices, baseline.vertices)

    for expected, actual in zip(
        baseline.normal_cones,
        optimized.normal_cones,
        strict=True,
    ):
        assert actual.active_facets == expected.active_facets
        assert np.allclose(actual.vertex, expected.vertex)
        assert np.allclose(actual.normals, expected.normals)

    for expected, actual in zip(
        baseline.normal_cones,
        jax_variant.normal_cones,
        strict=True,
    ):
        assert actual.active_facets == expected.active_facets
        assert np.allclose(actual.vertex, expected.vertex)
        assert np.allclose(actual.normals, expected.normals)


def _sorted_vertices(vertices: np.ndarray) -> np.ndarray:
    array = np.asarray(vertices, dtype=float)
    keys = np.lexsort(array.T[::-1])
    return array[keys]


def test_vertex_enumeration_variants_match() -> None:
    cube = hypercube(3)
    B, c = cube.halfspace_data()
    reference_vertices = vertices_from_halfspaces(B, c)
    optimized_vertices = vertices_from_halfspaces_optimized(B, c)
    jax_vertices = vertices_from_halfspaces_jax(B, c)

    expected = _sorted_vertices(reference_vertices)
    assert np.allclose(_sorted_vertices(optimized_vertices), expected)
    assert np.allclose(_sorted_vertices(jax_vertices), expected)


def test_polytope_combinatorics_cached_instances_are_reused() -> None:
    cube = hypercube(3)
    first = polytope_combinatorics(cube, use_cache=True)
    second = polytope_combinatorics(cube, use_cache=True)
    assert first is second


def test_halfspace_vertex_roundtrip() -> None:
    cube = hypercube(3)
    B, c = cube.halfspace_data()
    vertices = vertices_from_halfspaces(B, c)
    roundtrip_B, roundtrip_c = halfspaces_from_vertices(vertices)
    expected_B, expected_c = _sort_halfspaces(B, c)
    actual_B, actual_c = _sort_halfspaces(roundtrip_B, roundtrip_c)
    assert np.allclose(actual_B, expected_B)
    assert np.allclose(actual_c, expected_c)


def test_polytope_fingerprint_invariant_to_metadata() -> None:
    cube = hypercube(3)
    renamed = cube.with_metadata(name="renamed-cube")
    assert polytope_fingerprint(cube) == polytope_fingerprint(renamed)
