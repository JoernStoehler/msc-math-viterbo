"""Regression tests for half-space utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from viterbo.geometry.halfspaces import (
    enumerate_vertices,
    enumerate_vertices_jax,
    enumerate_vertices_optimized,
    remove_redundant_facets,
    remove_redundant_facets_jax,
    remove_redundant_facets_optimized,
    unit_square_halfspaces,
)


def _sorted_rows(array: np.ndarray) -> np.ndarray:
    return np.array(sorted(array.tolist()))


def _run_enumerator(
    enumerator: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    matrix, offsets = unit_square_halfspaces()
    vertices = enumerator(matrix, offsets)
    return np.asarray(vertices, dtype=float)


def test_reference_enumeration_matches_expected_square() -> None:
    vertices = _run_enumerator(enumerate_vertices)
    expected = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    assert vertices.shape == (4, 2)
    assert np.allclose(_sorted_rows(vertices), _sorted_rows(expected))


def test_optimized_enumerator_matches_reference() -> None:
    reference_vertices = _run_enumerator(enumerate_vertices)
    optimized_vertices = _run_enumerator(enumerate_vertices_optimized)
    assert np.allclose(_sorted_rows(reference_vertices), _sorted_rows(optimized_vertices))


def test_jax_enumerator_matches_reference() -> None:
    reference_vertices = _run_enumerator(enumerate_vertices)
    matrix, offsets = unit_square_halfspaces()
    jax_vertices = enumerate_vertices_jax(matrix, offsets)
    assert np.allclose(
        _sorted_rows(reference_vertices),
        _sorted_rows(np.asarray(jax_vertices, dtype=float)),
    )


def test_remove_redundant_facets_discards_duplicates_across_variants() -> None:
    matrix, offsets = unit_square_halfspaces()
    matrix = np.vstack((matrix, matrix[0:1]))
    offsets = np.concatenate((offsets, offsets[0:1]))

    reference_B, reference_c = remove_redundant_facets(matrix, offsets)
    optimized_B, optimized_c = remove_redundant_facets_optimized(matrix, offsets)
    jax_B, jax_c = remove_redundant_facets_jax(matrix, offsets)

    assert reference_B.shape == (4, 2)
    assert reference_c.shape == (4,)
    assert np.allclose(reference_B, optimized_B)
    assert np.allclose(reference_c, optimized_c)
    assert np.allclose(reference_B, np.asarray(jax_B))
    assert np.allclose(reference_c, np.asarray(jax_c))
