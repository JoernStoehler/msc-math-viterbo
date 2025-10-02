"""Tests for symplectic and convex-geometry helpers in :mod:`viterbo.core`."""

from __future__ import annotations

import numpy as np
import pytest

from viterbo.core import (
    minkowski_sum,
    standard_symplectic_matrix,
    support_function,
    symplectic_product,
)


def test_standard_symplectic_matrix_structure() -> None:
    matrix = standard_symplectic_matrix(4)
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(matrix, expected)


@pytest.mark.parametrize("dimension", [1, 3])
def test_standard_symplectic_matrix_requires_even_dimension(dimension: int) -> None:
    with pytest.raises(ValueError):
        standard_symplectic_matrix(dimension)


def test_symplectic_product_default_matrix() -> None:
    vector_a = np.array([1.0, 0.0, 0.0, 0.0])
    vector_b = np.array([0.0, 1.0, 0.0, 0.0])
    value = symplectic_product(vector_a, vector_b)
    assert value == pytest.approx(0.0)

    vector_c = np.array([1.0, 0.0, 0.0, 0.0])
    vector_d = np.array([0.0, 0.0, 1.0, 0.0])
    value_cd = symplectic_product(vector_c, vector_d)
    assert value_cd == pytest.approx(1.0)


def test_symplectic_product_custom_matrix() -> None:
    matrix = np.array([[0.0, 2.0], [-2.0, 0.0]])
    value = symplectic_product(np.array([1.0, 0.0]), np.array([0.0, 1.0]), matrix=matrix)
    assert value == pytest.approx(2.0)


def test_support_function_simplex() -> None:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    direction = np.array([1.0, 1.0])
    value = support_function(vertices, direction)
    assert value == pytest.approx(1.0)


def test_minkowski_sum_pairwise_vertices() -> None:
    square = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    segment = np.array([[0.0, 0.0], [0.0, 2.0]])
    result = minkowski_sum(square, segment)
    expected = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [0.0, 1.0],
            [0.0, 3.0],
            [1.0, 1.0],
            [1.0, 3.0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_support_function_validates_inputs() -> None:
    with pytest.raises(ValueError):
        support_function(np.empty((0, 2)), np.array([1.0, 0.0]))

    with pytest.raises(ValueError):
        support_function(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError):
        support_function(np.array([[1.0, 0.0]]), np.array([1.0, 0.0, 0.0]))


def test_minkowski_sum_validates_inputs() -> None:
    with pytest.raises(ValueError):
        minkowski_sum(np.empty((0, 2)), np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError):
        minkowski_sum(np.array([[1.0, 0.0]]), np.empty((0, 2)))

    with pytest.raises(ValueError):
        minkowski_sum(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
