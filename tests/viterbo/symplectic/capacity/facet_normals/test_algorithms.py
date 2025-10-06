"""Regression tests for the facet-normal EHZ capacity algorithms."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest import MonkeyPatch
from tests._utils.polytope_samples import load_polytope_instances

from viterbo.geometry.polytopes import (
    cross_polytope,
    hypercube,
    simplex_with_uniform_weights,
)
from viterbo.symplectic.capacity.facet_normals import (
    compute_ehz_capacity_fast,
    compute_ehz_capacity_reference,
)
from viterbo.symplectic.capacity.facet_normals import subset_utils as subset_utils
from viterbo.symplectic.core import standard_symplectic_matrix

_SMOKE_POLYTOPES = (
    simplex_with_uniform_weights(2, name="simplex-2d-smoke"),
)
_SMOKE_CAPACITY_CASES = tuple(
    pytest.param(*poly.halfspace_data(), id=poly.name) for poly in _SMOKE_POLYTOPES
)

_DEEP_STATIC_CAPACITY_CASES = (
    pytest.param(*hypercube(2, name="hypercube-2d-smoke").halfspace_data(), id="hypercube-2d", marks=(pytest.mark.deep,)),
    pytest.param(*cross_polytope(2, name="cross-polytope-2d-smoke").halfspace_data(), id="cross-polytope-2d", marks=(pytest.mark.deep,)),
)

_BASE_DATA = load_polytope_instances(variant_count=0)
_BASE_INSTANCES = list(_BASE_DATA[0])
_BASE_IDS = list(_BASE_DATA[1])


def _capacity_case(index: int) -> pytest.ParameterSet:
    B, c = _BASE_INSTANCES[index]
    identifier = _BASE_IDS[index]
    return pytest.param(B, c, id=identifier, marks=(pytest.mark.deep,))


_CAPACITY_CASES = _SMOKE_CAPACITY_CASES + _DEEP_STATIC_CAPACITY_CASES + tuple(
    _capacity_case(idx) for idx in range(len(_BASE_INSTANCES))
)


@pytest.fixture(name="subset_utils_close_records")
def fixture_subset_utils_close_records(
    monkeypatch: MonkeyPatch,
) -> dict[str, float | None]:
    """Patch ``subset_utils`` closeness helpers and capture tolerance arguments."""
    records: dict[str, float | None] = {
        "allclose": None,
        "allclose_rtol": None,
        "isclose": None,
        "isclose_rtol": None,
    }

    def fake_allclose(*args: object, **kwargs: object) -> bool:
        atol = kwargs.get("atol")
        rtol = kwargs.get("rtol")
        if atol is None and len(args) > 2:
            atol = args[2]
        if rtol is None and len(args) > 3:
            rtol = args[3]
        assert isinstance(atol, float)
        assert isinstance(rtol, float)
        records["allclose"] = atol
        records["allclose_rtol"] = rtol
        return True

    def fake_isclose(*args: object, **kwargs: object) -> bool:
        atol = kwargs.get("atol")
        rtol = kwargs.get("rtol")
        if atol is None and len(args) > 2:
            atol = args[2]
        if rtol is None and len(args) > 3:
            rtol = args[3]
        assert isinstance(atol, float)
        assert isinstance(rtol, float)
        records["isclose"] = atol
        records["isclose_rtol"] = rtol
        return True

    monkeypatch.setattr(subset_utils.np, "allclose", fake_allclose)
    monkeypatch.setattr(subset_utils.np, "isclose", fake_isclose)
    return records


@pytest.mark.goal_math
@pytest.mark.parametrize(("B", "c"), _CAPACITY_CASES)
def test_fast_matches_reference(B: np.ndarray, c: np.ndarray) -> None:
    """Reference and fast implementations should agree on sample polytopes."""
    try:
        reference_value = compute_ehz_capacity_reference(B, c)
    except ValueError as exc:
        with pytest.raises(ValueError) as caught:
            compute_ehz_capacity_fast(B, c)  # type: ignore[reportArgumentType]
        assert str(caught.value) == str(exc)
    else:
        fast_value = compute_ehz_capacity_fast(B, c)  # type: ignore[reportArgumentType]
        assert math.isclose(reference_value, fast_value, rel_tol=1e-10, abs_tol=1e-12)


@pytest.mark.goal_math
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
        base: float = compute_ehz_capacity_reference(B, c)
        transformed: float = compute_ehz_capacity_reference(transformed_B, c)  # type: ignore[reportArgumentType]
    except ValueError:
        pytest.skip("Reference algorithm is undefined for the chosen symmetric sample.")
    assert math.isclose(float(base), float(transformed), rel_tol=1e-10, abs_tol=1e-12)  # type: ignore[reportUnknownArgumentType]


@pytest.mark.goal_code
def test_rejects_odd_dimension() -> None:
    """Polytopes in odd ambient dimension should raise a ``ValueError``."""
    B = np.eye(3)
    c = np.ones(3)
    with pytest.raises(ValueError):
        compute_ehz_capacity_reference(B, c)
    with pytest.raises(ValueError):
        compute_ehz_capacity_fast(B, c)  # type: ignore[reportArgumentType]


@pytest.mark.goal_code
def test_prepare_subset_respects_tol(
    subset_utils_close_records: dict[str, float | None],
) -> None:
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
    subset = subset_utils.prepare_subset(  # type: ignore[reportArgumentType]
        B_matrix=B,  # type: ignore[reportArgumentType]
        c=c,  # type: ignore[reportArgumentType]
        indices=(0, 1, 2),
        J=J,
        tol=1e-6,  # type: ignore[reportArgumentType]
    )  # type: ignore[reportArgumentType]
    assert subset is not None

    allclose = subset_utils_close_records["allclose"]
    isclose_value = subset_utils_close_records["isclose"]
    allclose_rtol = subset_utils_close_records["allclose_rtol"]
    isclose_rtol = subset_utils_close_records["isclose_rtol"]

    assert allclose is not None
    assert isclose_value is not None
    assert allclose_rtol is not None
    assert isclose_rtol is not None

    assert math.isclose(allclose, 1e-6, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(isclose_value, 1e-6, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(allclose_rtol, 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(isclose_rtol, 0.0, rel_tol=0.0, abs_tol=1e-12)
