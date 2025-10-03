"""Regression tests for half-space utilities."""

from __future__ import annotations

import numpy as np

from viterbo.geometry.halfspaces import enumerate_vertices, remove_redundant_facets


def _square_halfspaces() -> tuple[np.ndarray, np.ndarray]:
    B_matrix = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    c = np.array([1.0, 1.0, 1.0, 1.0])
    return B_matrix, c


def test_enumerate_vertices_returns_expected_square() -> None:
    B_matrix, c = _square_halfspaces()
    vertices = enumerate_vertices(B_matrix, c)
    expected = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    assert vertices.shape == (4, 2)
    assert np.allclose(
        np.array(sorted(vertices.tolist())),
        np.array(sorted(expected.tolist())),
    )


def test_remove_redundant_facets_discards_duplicates() -> None:
    B_matrix, c = _square_halfspaces()
    B_matrix = np.vstack((B_matrix, B_matrix[0:1]))
    c = np.concatenate((c, c[0:1]))

    reduced_B, reduced_c = remove_redundant_facets(B_matrix, c)

    assert reduced_B.shape == (4, 2)
    assert reduced_c.shape == (4,)
    assert np.allclose(np.sort(reduced_c), np.sort(c[:4]))
