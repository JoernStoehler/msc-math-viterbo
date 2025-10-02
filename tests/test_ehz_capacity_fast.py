"""Tests ensuring the optimized EHZ capacity implementation matches the reference."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from viterbo import compute_ehz_capacity
from viterbo.ehz_fast import (
    _maximum_antisymmetric_order_value,
    compute_ehz_capacity_fast,
)
from viterbo.polytopes import catalog, random_transformations


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


def _instances() -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    rng = np.random.default_rng(2023)
    instances: list[tuple[np.ndarray, np.ndarray]] = []
    ids: list[str] = []
    for polytope in catalog():
        B, c = polytope.halfspace_data()
        instances.append((B, c))
        ids.append(polytope.name)

        variants = random_transformations(polytope, rng=rng, count=3)
        for index, (variant_B, variant_c) in enumerate(variants):
            instances.append((variant_B, variant_c))
            ids.append(f"{polytope.name}-variant-{index}")

    return instances, ids


_POLYTOPE_INSTANCES, _POLYTOPE_IDS = _instances()


@pytest.mark.parametrize(("B", "c"), _POLYTOPE_INSTANCES, ids=_POLYTOPE_IDS)
def test_fast_implementation_matches_reference(B: np.ndarray, c: np.ndarray) -> None:
    """The accelerated implementation matches the reference for diverse polytopes."""

    try:
        reference = compute_ehz_capacity(B, c)
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            compute_ehz_capacity_fast(B, c)
        assert str(caught.value) == str(error)
    else:
        optimized = compute_ehz_capacity_fast(B, c)
        assert np.isclose(reference, optimized, atol=1e-8)
