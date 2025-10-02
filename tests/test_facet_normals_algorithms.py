"""Regression tests for the facet-normal EHZ capacity algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from viterbo.algorithms import (
    compute_ehz_capacity_fast,
    compute_ehz_capacity_reference,
)

from ._polytope_samples import load_polytope_instances


_BASE_INSTANCES, _BASE_IDS = load_polytope_instances(variant_count=0)


@pytest.mark.parametrize(("B", "c"), _BASE_INSTANCES, ids=_BASE_IDS)
def test_fast_matches_reference(B: np.ndarray, c: np.ndarray) -> None:
    """Reference and fast implementations should agree on sample polytopes."""
    try:
        reference_value = compute_ehz_capacity_reference(B, c)
    except ValueError as exc:
        with pytest.raises(ValueError) as caught:
            compute_ehz_capacity_fast(B, c)
        assert str(caught.value) == str(exc)
    else:
        fast_value = compute_ehz_capacity_fast(B, c)
        assert pytest.approx(reference_value, rel=1e-10, abs=1e-12) == fast_value


def test_symplectic_invariance_square() -> None:
    """A linear symplectic change of coordinates preserves the capacity."""
    B, c = _BASE_INSTANCES[0]
    dimension = B.shape[1]
    theta = np.pi / 6.0
    block = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    transform = np.block(
        [
            [block if i == j else np.zeros_like(block) for j in range(dimension // 2)]
            for i in range(dimension // 2)
        ]
    )
    transformed_B = B @ np.linalg.inv(transform)
    try:
        base = compute_ehz_capacity_reference(B, c)
        transformed = compute_ehz_capacity_reference(transformed_B, c)
    except ValueError:
        pytest.skip("Reference algorithm is undefined for the chosen symmetric sample.")
    assert pytest.approx(base, rel=1e-10, abs=1e-12) == transformed


def test_rejects_odd_dimension() -> None:
    """Polytopes in odd ambient dimension should raise a ``ValueError``."""
    B = np.eye(3)
    c = np.ones(3)
    with pytest.raises(ValueError):
        compute_ehz_capacity_reference(B, c)
    with pytest.raises(ValueError):
        compute_ehz_capacity_fast(B, c)
