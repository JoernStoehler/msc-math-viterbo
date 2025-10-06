"""Unit tests for :mod:`viterbo.symplectic.core` (JAX-first)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from viterbo.symplectic.core import (
    ZERO_TOLERANCE,
    minkowski_sum,
    normalize_vector,
    standard_symplectic_matrix,
    support_function,
    symplectic_product,
)


@pytest.mark.goal_math
def test_standard_symplectic_matrix_structure() -> None:
    """The canonical symplectic form has the expected block off-diagonal structure."""
    matrix = standard_symplectic_matrix(4)
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(np.asarray(matrix), expected)


@pytest.mark.goal_code
@pytest.mark.parametrize("dimension", [1, 3])
def test_standard_symplectic_matrix_requires_even_dimension(dimension: int) -> None:
    """Odd-dimensional requests raise errors because the symplectic matrix is 2n×2n."""
    with pytest.raises(ValueError):
        standard_symplectic_matrix(dimension)


@pytest.mark.goal_math
def test_symplectic_product_default_matrix() -> None:
    """Symplectic products respect the canonical pairing on standard basis vectors."""
    vector_a = jnp.array([1.0, 0.0, 0.0, 0.0])
    vector_b = jnp.array([0.0, 1.0, 0.0, 0.0])
    value = symplectic_product(vector_a, vector_b)
    assert math.isclose(value, 0.0, rel_tol=1e-12, abs_tol=0.0)

    vector_c = jnp.array([1.0, 0.0, 0.0, 0.0])
    vector_d = jnp.array([0.0, 0.0, 1.0, 0.0])
    value_cd = symplectic_product(vector_c, vector_d)
    assert math.isclose(value_cd, 1.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_math
def test_symplectic_product_custom_matrix() -> None:
    """Supplying a custom symplectic matrix rescales the pairing accordingly."""
    matrix = jnp.array([[0.0, 2.0], [-2.0, 0.0]])
    value = symplectic_product(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), matrix=matrix)
    assert math.isclose(value, 2.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_math
def test_support_function_simplex() -> None:
    """Support function of a simplex equals the maximum dot product along a direction."""
    vertices = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    direction = jnp.array([1.0, 1.0])
    value = support_function(vertices, direction)
    assert math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_math
def test_minkowski_sum_pairwise_vertices() -> None:
    """Minkowski sums enumerate pairwise vertex additions for convex polytopes."""
    square = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    segment = jnp.array([[0.0, 0.0], [0.0, 2.0]])
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
    np.testing.assert_array_equal(np.asarray(result), expected)


@pytest.mark.goal_code
def test_support_function_validates_inputs() -> None:
    """Support function rejects empty vertices, rank mismatches, and invalid directions."""
    with pytest.raises(ValueError):
        support_function(jnp.empty((0, 2)), jnp.array([1.0, 0.0]))

    with pytest.raises(ValueError):
        support_function(jnp.array([[1.0, 0.0]]), jnp.array([[1.0, 0.0]]))

    with pytest.raises(ValueError):
        support_function(jnp.array([[1.0, 0.0]]), jnp.array([1.0, 0.0, 0.0]))


@pytest.mark.goal_code
def test_minkowski_sum_validates_inputs() -> None:
    """Minkowski sum enforces non-empty operands with matching dimensions."""
    with pytest.raises(ValueError):
        minkowski_sum(jnp.empty((0, 2)), jnp.array([[1.0, 0.0]]))

    with pytest.raises(ValueError):
        minkowski_sum(jnp.array([[1.0, 0.0]]), jnp.empty((0, 2)))

    with pytest.raises(ValueError):
        minkowski_sum(jnp.array([[1.0, 0.0]]), jnp.array([[1.0, 0.0, 0.0]]))


@pytest.mark.goal_math
def test_normalize_vector_unit_length() -> None:
    """Normalising a vector scales it to unit length under the Euclidean norm."""
    vector = jnp.array([3.0, 4.0])
    normalized = normalize_vector(vector)
    norm = float(jnp.linalg.norm(jnp.asarray(normalized)))
    assert math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_code
def test_normalize_vector_zero_vector_raises() -> None:
    """Normalisation rejects the zero vector because it has undefined direction."""
    zero = jnp.zeros(3)
    with pytest.raises(ValueError):
        normalize_vector(zero)


@pytest.mark.goal_code
def test_normalize_vector_accepts_list_input() -> None:
    """Lists are coerced to arrays before normalisation to support ergonomic APIs."""
    values = [3.0, 4.0, 12.0]
    normalized = normalize_vector(values)  # type: ignore[reportArgumentType]
    norm = float(jnp.linalg.norm(jnp.asarray(normalized)))
    assert math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_code
def test_zero_tolerance_reasonable() -> None:
    """The default numerical tolerance is positive yet comfortably small."""
    assert 0.0 < ZERO_TOLERANCE < 1e-6
