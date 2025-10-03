"""Regression tests for the facet-normal EHZ capacity algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import MonkeyPatch

import viterbo.symplectic.capacity_algorithms.facet_normals_reference as reference
from tests.geometry._polytope_samples import load_polytope_instances
from viterbo.symplectic.capacity_algorithms import (
    compute_ehz_capacity_fast,
    compute_ehz_capacity_reference,
)
from viterbo.symplectic.core import standard_symplectic_matrix

_BASE_DATA = load_polytope_instances(variant_count=0)
_BASE_INSTANCES = _BASE_DATA[0]
_BASE_IDS = _BASE_DATA[1]


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
        assert (
            pytest.approx(reference_value, rel=1e-10, abs=1e-12)  # type: ignore[reportUnknownMemberType]  # Pytest stubs incomplete; TODO: refine types
            == fast_value
        )


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
    assert (
        pytest.approx(base, rel=1e-10, abs=1e-12)  # type: ignore[reportUnknownMemberType]  # Pytest stubs incomplete; TODO: refine types
        == transformed
    )


def test_rejects_odd_dimension() -> None:
    """Polytopes in odd ambient dimension should raise a ``ValueError``."""
    B = np.eye(3)
    c = np.ones(3)
    with pytest.raises(ValueError):
        compute_ehz_capacity_reference(B, c)
    with pytest.raises(ValueError):
        compute_ehz_capacity_fast(B, c)


def test_prepare_subset_respects_tol(monkeypatch: MonkeyPatch) -> None:
    """Internal subset feasibility checks should use the caller-provided tol."""
    B = np.array(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
        ]
    )
    c = np.array([0.0, 0.0, 1.0])
    J = standard_symplectic_matrix(2)
    records: dict[str, float | None] = {
        "allclose": None,
        "allclose_rtol": None,
        "isclose": None,
        "isclose_rtol": None,
    }

    def fake_allclose(*args: object, **kwargs: object) -> bool:
        atol = kwargs.get("atol")
        rtol = kwargs.get("rtol")
        assert isinstance(atol, float)
        assert isinstance(rtol, float)
        records["allclose"] = atol
        records["allclose_rtol"] = rtol
        return True

    def fake_isclose(*args: object, **kwargs: object) -> bool:
        atol = kwargs.get("atol")
        rtol = kwargs.get("rtol")
        assert isinstance(atol, float)
        assert isinstance(rtol, float)
        records["isclose"] = atol
        records["isclose_rtol"] = rtol
        return True

    monkeypatch.setattr(reference.np, "allclose", fake_allclose)
    monkeypatch.setattr(reference.np, "isclose", fake_isclose)

    subset = reference._prepare_subset(B=B, c=c, indices=(0, 1, 2), J=J, tol=1e-6)
    assert subset is not None
    assert records["allclose"] == pytest.approx(1e-6)
    assert records["isclose"] == pytest.approx(1e-6)
    assert records["allclose_rtol"] == pytest.approx(0.0)
    assert records["isclose_rtol"] == pytest.approx(0.0)
