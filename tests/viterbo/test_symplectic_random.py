"""Random symplectic matrices and invariance checks (smoke)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from viterbo.datasets import builders as polytopes
from viterbo.math.capacity.facet_normals import ehz_capacity_reference_facet_normals
from viterbo.math import symplectic


@pytest.mark.goal_code
@pytest.mark.smoke
def test_random_symplectic_is_symplectic() -> None:
    """M^T J M == J within tolerance for sampled matrices."""
    key = jax.random.PRNGKey(0)
    dim = 4
    M = symplectic.random_symplectic_matrix(key, dim)
    J = symplectic.standard_symplectic_matrix(dim)
    lhs = M.T @ J @ M
    assert jnp.allclose(lhs, J, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_capacity_invariance_under_random_symplectic_4d() -> None:
    """c_EHZ invariant under linear symplectic transformations (4D)."""
    key = jax.random.PRNGKey(123)
    V = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0],
            [0.0, 0.0, 0.0, 0.8],
        ],
        dtype=jnp.float64,
    )
    P = polytopes.build_from_vertices(V)
    c0 = ehz_capacity_reference_facet_normals(P.normals, P.offsets)
    M = symplectic.random_symplectic_matrix(key, 4)
    P2 = polytopes.build_from_vertices(V @ M.T)
    c1 = ehz_capacity_reference_facet_normals(P2.normals, P2.offsets)
    assert jnp.isclose(c0, c1, rtol=1e-9, atol=1e-12)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_standard_symplectic_matrix_structure() -> None:
    """Standard symplectic matrix has the canonical block off-diagonal form."""

    matrix = symplectic.standard_symplectic_matrix(4)
    expected = jnp.asarray(
        [
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    assert jnp.array_equal(matrix, expected)


@pytest.mark.goal_code
@pytest.mark.smoke
@pytest.mark.parametrize("dimension", [1, 3])
def test_standard_symplectic_matrix_requires_even_dimension(dimension: int) -> None:
    """Odd-dimensional requests raise a ValueError."""

    with pytest.raises(ValueError, match="even"):
        symplectic.standard_symplectic_matrix(dimension)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_symplectic_product_default_matrix() -> None:
    """Symplectic product matches the canonical bilinear form."""

    vector_a = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    vector_b = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    value = symplectic.symplectic_product(vector_a, vector_b)
    assert value == pytest.approx(-1.0, rel=0.0, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_symplectic_product_custom_matrix() -> None:
    """Custom symplectic matrices rescale the bilinear pairing."""

    matrix = jnp.asarray(
        [
            [0.0, 2.0],
            [-2.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    vector_c = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    vector_d = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    value = symplectic.symplectic_product(vector_c, vector_d, matrix=matrix)
    assert value == pytest.approx(2.0, rel=0.0, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_support_function_over_vertices() -> None:
    """Support function attains the maximum inner product over vertices."""

    vertices = jnp.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    direction = jnp.asarray([0.5, 2.0], dtype=jnp.float64)
    value = symplectic.support_function(vertices, direction)
    assert value == pytest.approx(2.0, rel=0.0, abs=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_minkowski_sum_cartesian_product() -> None:
    """Minkowski sum enumerates all pairwise vertex sums."""

    first = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    second = jnp.asarray([[0.0, 1.0], [0.0, -1.0]], dtype=jnp.float64)
    summed = symplectic.minkowski_sum(first, second)
    expected = jnp.asarray(
        [
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    assert jnp.array_equal(summed, expected)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_normalize_vector_rejects_zero_vector() -> None:
    """Normalization rejects inputs whose norm falls below ZERO_TOLERANCE."""

    with pytest.raises(ValueError, match="near-zero"):
        symplectic.normalize_vector(jnp.zeros((2,), dtype=jnp.float64))
