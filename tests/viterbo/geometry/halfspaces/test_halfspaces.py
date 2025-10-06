"""Regression tests for half-space utilities (JAX-first)."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from viterbo.geometry.halfspaces import (
    enumerate_vertices_fast,
    enumerate_vertices_reference,
    remove_redundant_facets_fast,
    remove_redundant_facets_reference,
    unit_square_halfspaces,
)


def _sorted_rows(array: object) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    return np.array(sorted(arr.tolist()))


def _run_enumerator(
    enumerator: Callable[[Float[Array, " m n"], Float[Array, " m"]], Float[Array, " k n"]],
) -> np.ndarray:
    matrix, offsets = unit_square_halfspaces()
    vertices = enumerator(matrix, offsets)
    return np.asarray(vertices, dtype=float)

@pytest.mark.goal_math
def test_reference_enumeration_matches_expected_square() -> None:
    """Reference enumeration returns the vertices of the canonical unit square."""
    vertices = _run_enumerator(enumerate_vertices_reference)
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


@pytest.mark.goal_math
def test_fast_enumerator_matches_reference() -> None:
    """The fast vertex enumerator matches the reference implementation on the square."""
    ref = _run_enumerator(enumerate_vertices_reference)
    fast = _run_enumerator(enumerate_vertices_fast)
    expected = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )
    assert np.allclose(_sorted_rows(ref), _sorted_rows(expected))
    assert np.allclose(_sorted_rows(fast), _sorted_rows(expected))


@pytest.mark.goal_code
def test_remove_redundant_facets_discards_duplicates() -> None:
    """Redundant facets are removed consistently by reference and fast implementations."""
    matrix, offsets = unit_square_halfspaces()
    matrix = jnp.vstack((matrix, matrix[0:1]))
    offsets = jnp.concatenate((offsets, offsets[0:1]))
    ref_B, ref_c = remove_redundant_facets_reference(matrix, offsets)
    fast_B, fast_c = remove_redundant_facets_fast(matrix, offsets)
    assert ref_B.shape == (4, 2)
    assert ref_c.shape == (4,)
    assert np.allclose(np.asarray(ref_B), np.asarray(fast_B))
    assert np.allclose(np.asarray(ref_c), np.asarray(fast_c))
