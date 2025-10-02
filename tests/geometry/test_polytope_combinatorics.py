"""Tests for cached combinatorial data of polytopes."""

from __future__ import annotations

import numpy as np

from viterbo.geometry.polytopes import (
    PolytopeCombinatorics,
    halfspaces_from_vertices,
    hypercube,
    polytope_combinatorics,
    polytope_fingerprint,
    vertices_from_halfspaces,
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
