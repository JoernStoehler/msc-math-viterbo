from __future__ import annotations

import math
from typing import Final

import pytest

from viterbo.symplectic.capacity.support_relaxation.reference import (
    compute_support_relaxation_capacity_reference,
)

_: Final = pytest.importorskip("cvxpy")


def _assert_monotone(history: list[float]) -> None:
    for left, right in zip(history, history[1:]):
        assert left + 1e-12 >= right


def test_reference_solver_converges_on_unit_disk(unit_disk_vertices) -> None:
    result = compute_support_relaxation_capacity_reference(
        unit_disk_vertices,
        grid_density=15,
        smoothing_parameters=(1.2, 0.9, 0.6, 0.3, 0.1, 0.0),
        tolerance_sequence=(1e-5, 1e-6, 1e-6),
    )
    capacities = [diagnostic.candidate_capacity for diagnostic in result.history]
    _assert_monotone(capacities)
    assert capacities[-1] == pytest.approx(result.capacity_upper_bound)
    assert result.capacity_upper_bound == pytest.approx(math.pi, rel=5e-2)


def test_reference_solver_translation_invariant(unit_disk_vertices) -> None:
    base = compute_support_relaxation_capacity_reference(unit_disk_vertices)
    translated = compute_support_relaxation_capacity_reference(
        unit_disk_vertices + 0.25,
    )
    assert translated.capacity_upper_bound == pytest.approx(
        base.capacity_upper_bound, rel=1e-9
    )
