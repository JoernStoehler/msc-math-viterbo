"""Tests ensuring the optimized EHZ capacity implementation matches the reference."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from viterbo import compute_ehz_capacity
from viterbo.ehz_fast import (
    _maximum_antisymmetric_order_value,
    compute_ehz_capacity_fast,
)


def _simplex_like_polytope_data(dimension: int = 4) -> tuple[np.ndarray, np.ndarray]:
    B = np.eye(dimension)
    extra = -np.ones((1, dimension))
    B = np.vstack((B, extra))
    c = np.ones(dimension + 1)
    c[-1] = dimension / 2
    return B, c


def _simplex_with_extra_facet_data() -> tuple[np.ndarray, np.ndarray]:
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


def _generate_variants(
    B: np.ndarray,
    c: np.ndarray,
    *,
    rng: np.random.Generator,
    count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dimension = B.shape[1]
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(count):
        random_matrix = rng.normal(size=(dimension, dimension))
        q, _ = np.linalg.qr(random_matrix)
        scales = rng.uniform(0.6, 1.4, size=dimension)
        transform = q @ np.diag(scales)
        transformed_B = B @ transform
        translation = rng.normal(scale=0.3, size=dimension)
        transformed_c = c + transformed_B @ translation
        results.append((transformed_B, transformed_c))
    return results


def test_dynamic_program_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(5, 5))
    weights = weights - weights.T

    brute = -np.inf
    for order in permutations(range(5)):
        total = 0.0
        for i in range(1, 5):
            idx_i = order[i]
            for j in range(i):
                idx_j = order[j]
                total += weights[idx_i, idx_j]
        brute = max(brute, total)

    dp_value = _maximum_antisymmetric_order_value(weights)
    assert np.isclose(dp_value, brute)


def test_fast_implementation_matches_reference() -> None:
    rng = np.random.default_rng(2023)
    base_polytopes = [
        _simplex_like_polytope_data(),
        _simplex_with_extra_facet_data(),
        _simplex_like_polytope_data(6),
    ]

    instances: list[tuple[np.ndarray, np.ndarray]] = []
    for B, c in base_polytopes:
        instances.append((B, c))
        instances.extend(_generate_variants(B, c, rng=rng, count=3))

    for B, c in instances:
        reference = compute_ehz_capacity(B, c)
        optimized = compute_ehz_capacity_fast(B, c)
        assert np.isclose(reference, optimized, atol=1e-8)
