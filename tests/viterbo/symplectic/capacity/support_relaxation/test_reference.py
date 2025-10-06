from __future__ import annotations

import math

import pytest

pytest.importorskip("cvxpy")

from viterbo.symplectic.capacity.support_relaxation.reference import (
    compute_support_relaxation_capacity_reference,
)


def _assert_monotone(history: list[float]) -> None:
    for left, right in zip(history, history[1:]):
        assert left + 1e-12 >= right


@pytest.mark.goal_math
def test_reference_solver_converges_on_unit_disk(unit_disk_vertices) -> None:
    """Reference solver's bound sequence decreases and approaches pi."""
    result = compute_support_relaxation_capacity_reference(
        unit_disk_vertices,
        grid_density=15,
        smoothing_parameters=(1.2, 0.9, 0.6, 0.3, 0.1, 0.0),
        smoothing_method="convex",
        tolerance_sequence=(1e-5, 1e-6, 1e-6),
    )
    capacities = [diagnostic.candidate_capacity for diagnostic in result.history]
    _assert_monotone(capacities)
    assert capacities[-1] == pytest.approx(result.capacity_upper_bound)
    assert result.capacity_upper_bound == pytest.approx(math.pi, rel=5e-2)


@pytest.mark.goal_math
def test_reference_solver_translation_invariant(unit_disk_vertices) -> None:
    """Capacity estimate is invariant under translations of vertices."""
    base = compute_support_relaxation_capacity_reference(unit_disk_vertices)
    translated = compute_support_relaxation_capacity_reference(
        unit_disk_vertices + 0.25,
    )
    assert translated.capacity_upper_bound == pytest.approx(base.capacity_upper_bound, rel=1e-9)


@pytest.mark.goal_math
def test_reference_solver_softmax_schedule(unit_disk_vertices) -> None:
    """Softmax smoothing schedule computes a positive capacity bound."""
    result = compute_support_relaxation_capacity_reference(
        unit_disk_vertices,
        smoothing_parameters=(0.6, 0.3, 0.0),
        smoothing_method="softmax",
    )
    assert result.capacity_upper_bound > 0.0
