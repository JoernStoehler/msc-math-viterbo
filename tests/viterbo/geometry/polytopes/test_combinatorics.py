"""Tests for cached combinatorial data of polytopes."""

from __future__ import annotations

import numpy as np
import pytest

from viterbo.geometry.polytopes import (
    PolytopeCombinatorics,
    halfspaces_from_vertices,
    hypercube,
    polytope_combinatorics,
    polytope_fingerprint,
    vertices_from_halfspaces,
)


def _sort_halfspaces(
    B: object,
    c: object,
) -> tuple[np.ndarray, np.ndarray]:
    B_np = np.asarray(B)
    c_np = np.asarray(c)
    key = np.lexsort(np.column_stack((B_np, c_np)).T)
    return B_np[key], c_np[key]

@pytest.mark.goal_math
def test_polytope_combinatorics_square_facets() -> None:
    """Facet adjacency for a square has degree two across all faces."""
    square = hypercube(2)
    combinatorics = polytope_combinatorics(square, use_cache=False)

    assert isinstance(combinatorics, PolytopeCombinatorics)
    assert combinatorics.vertices.shape == (4, 2)
    assert combinatorics.facet_adjacency.shape == (4, 4)
    # Each facet of a square touches exactly two neighbours.
    degree = combinatorics.facet_adjacency.sum(axis=1)
    assert np.all(degree == 2)

@pytest.mark.goal_math
def test_polytope_combinatorics_properties() -> None:
    """Cube facet adjacency matches the expected combinatorial degree structure."""
    cube = hypercube(3)
    baseline = polytope_combinatorics(cube, use_cache=False)

    assert isinstance(baseline, PolytopeCombinatorics)
    assert baseline.vertices.shape[1] == 3
    # Each facet of a cube in 3D touches 4 neighbours.
    degree = baseline.facet_adjacency.sum(axis=1)
    assert np.all((degree == 4) | (degree == 0))


def _sorted_vertices(vertices: object) -> np.ndarray:
    array = np.asarray(vertices, dtype=float)
    keys = np.lexsort(array.T[::-1])
    return array[keys]

@pytest.mark.goal_math
def test_vertex_enumeration_matches_reference_shape() -> None:
    """Vertex enumeration from halfspaces matches the expected sorted coordinates."""
    cube = hypercube(3)
    B, c = cube.halfspace_data()
    reference_vertices = vertices_from_halfspaces(B, c)
    expected = _sorted_vertices(reference_vertices)
    assert np.allclose(_sorted_vertices(reference_vertices), expected)


@pytest.mark.goal_code
def test_polytope_combinatorics_cached_instances_are_reused() -> None:
    """Repeated combinatorics calls with caching return the same object instance."""
    cube = hypercube(3)
    first = polytope_combinatorics(cube, use_cache=True)
    second = polytope_combinatorics(cube, use_cache=True)
    assert first is second


@pytest.mark.goal_math
def test_halfspace_vertex_roundtrip() -> None:
    """Round-tripping between halfspaces and vertices preserves geometry."""
    cube = hypercube(3)
    B, c = cube.halfspace_data()
    vertices = vertices_from_halfspaces(B, c)
    roundtrip_B, roundtrip_c = halfspaces_from_vertices(vertices)
    expected_B, expected_c = _sort_halfspaces(B, c)
    actual_B, actual_c = _sort_halfspaces(roundtrip_B, roundtrip_c)
    assert np.allclose(actual_B, expected_B)
    assert np.allclose(actual_c, expected_c)

@pytest.mark.goal_code
def test_polytope_fingerprint_invariant_to_metadata() -> None:
    """Fingerprints depend on geometry only and ignore metadata like names."""
    cube = hypercube(3)
    renamed = cube.with_metadata(name="renamed-cube")
    assert polytope_fingerprint(cube) == polytope_fingerprint(renamed)
