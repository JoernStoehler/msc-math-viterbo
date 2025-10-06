"""Tests ensuring the optimized EHZ capacity implementation matches the reference."""

from __future__ import annotations

from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._utils.polytope_samples import load_polytope_instances
from viterbo.geometry.polytopes import (
    cross_polytope,
    hypercube,
    simplex_with_uniform_weights,
)
from viterbo.symplectic.capacity import compute_ehz_capacity_reference
from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast,
)
from viterbo.symplectic.capacity.facet_normals.subset_utils import (
    maximum_antisymmetric_order_value,
)


def test_dynamic_program_matches_bruteforce() -> None:
    key = jax.random.PRNGKey(0)
    weights = jax.random.normal(key, (5, 5), dtype=jnp.float64)
    weights = weights - weights.T
    weights_np = np.asarray(weights)

    brute = -np.inf
    for order in permutations(range(5)):
        total = 0.0
        for i in range(1, 5):
            idx_i = order[i]
            for j in range(i):
                idx_j = order[j]
                total += weights_np[idx_i, idx_j]
        brute = max(brute, total)

    dp_value = maximum_antisymmetric_order_value(weights_np)  # type: ignore[reportArgumentType]
    assert np.isclose(dp_value, brute, atol=1e-12)


_SMOKE_POLYTOPES = (
    simplex_with_uniform_weights(2, name="simplex-2d-smoke"),
)
_SMOKE_FAST_CASES = tuple(
    pytest.param(*poly.halfspace_data(), id=poly.name) for poly in _SMOKE_POLYTOPES
)

_DEEP_STATIC_FAST_CASES = (
    pytest.param(*hypercube(2, name="hypercube-2d-smoke").halfspace_data(), id="hypercube-2d", marks=(pytest.mark.deep,)),
    pytest.param(*cross_polytope(2, name="cross-polytope-2d-smoke").halfspace_data(), id="cross-polytope-2d", marks=(pytest.mark.deep,)),
)

_POLYTOPE_DATA = load_polytope_instances()
_POLYTOPE_INSTANCES = list(_POLYTOPE_DATA[0])
_POLYTOPE_IDS = list(_POLYTOPE_DATA[1])


def _fast_case(index: int) -> pytest.ParameterSet:
    B, c = _POLYTOPE_INSTANCES[index]
    identifier = _POLYTOPE_IDS[index]
    return pytest.param(B, c, id=identifier, marks=(pytest.mark.deep,))


_FAST_CASES = _SMOKE_FAST_CASES + _DEEP_STATIC_FAST_CASES + tuple(
    _fast_case(idx) for idx in range(len(_POLYTOPE_INSTANCES))
)


@pytest.mark.parametrize(("B", "c"), _FAST_CASES)
def test_fast_implementation_matches_reference(B: np.ndarray, c: np.ndarray) -> None:
    """The accelerated implementation matches the reference for diverse polytopes."""

    try:
        reference = compute_ehz_capacity_reference(B, c)  # type: ignore[reportArgumentType]
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            compute_ehz_capacity_fast(B, c)  # type: ignore[reportArgumentType]
        assert str(caught.value) == str(error)
    else:
        optimized = compute_ehz_capacity_fast(B, c)  # type: ignore[reportArgumentType]
        assert np.isclose(reference, optimized, atol=1e-8)
