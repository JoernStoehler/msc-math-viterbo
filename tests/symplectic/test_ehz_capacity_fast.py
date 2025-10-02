"""Tests ensuring the optimized EHZ capacity implementation matches the reference."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from tests.geometry._polytope_samples import load_polytope_instances
from viterbo.symplectic.capacity import compute_ehz_capacity
from viterbo.symplectic.capacity_algorithms.facet_normals_fast import (
    _maximum_antisymmetric_order_value,
    compute_ehz_capacity_fast,
)


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


_POLYTOPE_DATA = load_polytope_instances()
_POLYTOPE_INSTANCES = _POLYTOPE_DATA[0]
_POLYTOPE_IDS = _POLYTOPE_DATA[1]


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
