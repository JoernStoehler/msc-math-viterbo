"""Tests for the combinatorial EHZ capacity implementation."""

from __future__ import annotations

import numpy as np

from viterbo import compute_ehz_capacity, standard_symplectic_matrix


def _simplex_like_polytope_data() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(B, c)`` for a non-degenerate 4D simplex with uniform weights."""

    B = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    c = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    return B, c


def _simplex_with_extra_facet_data() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(B, c)`` for a 4D simplex truncated by an additional facet."""

    B = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    c = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.2])
    return B, c


def _haim_kislev_candidate(
    B: np.ndarray, c: np.ndarray, *, subset: tuple[int, ...], order: tuple[int, ...]
) -> float:
    """Evaluate the Haim–Kislev action for a fixed facet subset and order."""

    J = standard_symplectic_matrix(B.shape[1])
    rows = np.array(subset)
    B_subset = B[rows]
    c_subset = c[rows]
    m = len(subset)

    system = np.zeros((m, m))
    system[0, :] = c_subset
    system[1:, :] = B_subset.T

    rhs = np.zeros(m)
    rhs[0] = 1.0

    beta = np.linalg.solve(system, rhs)
    symplectic_products = (B_subset @ J) @ B_subset.T

    total = 0.0
    for i in range(1, m):
        idx_i = order[i]
        weight_i = beta[idx_i]
        if weight_i <= 0.0:
            continue
        row = symplectic_products[idx_i]
        for j in range(i):
            idx_j = order[j]
            weight_j = beta[idx_j]
            if weight_j <= 0.0:
                continue
            total += weight_i * weight_j * row[idx_j]

    if total <= 0.0:
        msg = "Facet ordering yielded a non-positive action."
        raise ValueError(msg)

    return 0.5 / total


def test_cube_cross_polytope_capacity_matches_theory() -> None:
    r"""The simplex-like model has capacity nine by direct computation."""

    B, c = _simplex_like_polytope_data()
    capacity = compute_ehz_capacity(B, c)
    assert np.isclose(capacity, 9.0, atol=1e-9)


def test_capacity_scales_quadratically_under_dilation() -> None:
    r"""Scaling the polytope dilates the capacity by the square factor."""

    B, c = _simplex_like_polytope_data()
    base_capacity = compute_ehz_capacity(B, c)

    scale = 1.5
    scaled_capacity = compute_ehz_capacity(B, scale * c)

    assert np.isclose(scaled_capacity, (scale**2) * base_capacity, atol=1e-8)


def test_capacity_is_translation_invariant() -> None:
    """Rigid translations of the polytope leave ``c_EHZ`` unchanged."""

    B, c = _simplex_like_polytope_data()
    base_capacity = compute_ehz_capacity(B, c)

    translation = np.array([0.3, -0.2, 0.1, -0.4])
    translated_c = c + B @ translation

    translated_capacity = compute_ehz_capacity(B, translated_c)

    assert np.isclose(translated_capacity, base_capacity, atol=1e-9)


def test_truncated_simplex_matches_known_subset_action() -> None:
    """Adding an extra facet leaves the optimal action unchanged."""

    B, c = _simplex_with_extra_facet_data()
    capacity = compute_ehz_capacity(B, c)

    expected = _haim_kislev_candidate(
        B,
        c,
        subset=(0, 1, 2, 3, 4),
        order=(2, 0, 4, 3, 1),
    )

    assert np.isclose(capacity, expected, atol=1e-9)
