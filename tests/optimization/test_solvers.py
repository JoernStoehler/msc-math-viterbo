"""Unit tests for the solver abstraction layer."""

from __future__ import annotations

import numpy as np
import pytest

from viterbo.optimization.solvers import (
    LinearProgram,
    ScipyLinearProgramBackend,
    solve_linear_program,
)


def test_linear_program_validation_rejects_mismatched_rhs() -> None:
    objective = np.ones(2)
    with pytest.raises(ValueError):
        LinearProgram(objective=objective, lhs_ineq=np.ones((1, 2)), rhs_ineq=np.ones(2))


def test_scipy_backend_solves_simple_problem() -> None:
    problem = LinearProgram(
        objective=np.array([1.0, 0.0]),
        lhs_ineq=np.array([[-1.0, 0.0], [0.0, -1.0]]),
        rhs_ineq=np.zeros(2),
        lhs_eq=np.array([[1.0, 1.0]]),
        rhs_eq=np.array([1.0]),
    )
    backend = ScipyLinearProgramBackend()
    solution = backend.solve(problem)
    assert solution.status == "optimal"
    assert np.allclose(solution.x, np.array([0.0, 1.0]), atol=1e-8)
    assert pytest.approx(0.0, abs=1e-8) == solution.objective_value


def test_solve_linear_program_uses_default_backend() -> None:
    problem = LinearProgram(
        objective=np.array([0.0, 1.0]),
        lhs_ineq=np.array([[-1.0, 0.0], [0.0, -1.0]]),
        rhs_ineq=np.zeros(2),
        lhs_eq=np.array([[1.0, 1.0]]),
        rhs_eq=np.array([1.0]),
    )
    solution = solve_linear_program(problem)
    assert solution.status == "optimal"
    assert np.allclose(solution.x, np.array([1.0, 0.0]), atol=1e-8)
