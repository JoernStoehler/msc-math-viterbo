from __future__ import annotations

import math

import pytest

from viterbo.symplectic.capacity.support_relaxation.fast import (
    SupportRelaxationDiagnostics,
    compute_support_relaxation_capacity_fast,
)


def _assert_monotone(history: list[float]) -> None:
    for left, right in zip(history, history[1:]):
        assert left + 1e-12 >= right


def test_fast_solver_converges_on_unit_disk(unit_disk_vertices) -> None:
    logs: list[SupportRelaxationDiagnostics] = []
    result = compute_support_relaxation_capacity_fast(
        unit_disk_vertices,
        initial_density=9,
        refinement_steps=2,
        smoothing_parameters=(0.9, 0.6, 0.3, 0.1, 0.0),
        tolerance_sequence=(1e-3, 1e-4, 1e-5),
        log_callback=logs.append,
        jit_compile=False,
    )
    assert logs, "log_callback should record diagnostics"
    capacities = [diagnostic.candidate_capacity for diagnostic in result.history]
    _assert_monotone(capacities)
    assert capacities[-1] == pytest.approx(result.capacity_upper_bound)
    assert result.capacity_upper_bound == pytest.approx(math.pi, rel=7e-2)


def test_fast_solver_translation_invariant(unit_disk_vertices) -> None:
    base = compute_support_relaxation_capacity_fast(unit_disk_vertices, jit_compile=False)
    translated = compute_support_relaxation_capacity_fast(
        unit_disk_vertices + 0.75,
        jit_compile=False,
    )
    assert translated.capacity_upper_bound == pytest.approx(
        base.capacity_upper_bound, rel=1e-9
    )
