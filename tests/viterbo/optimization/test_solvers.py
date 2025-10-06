"""Unit tests for the solver abstraction layer."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from viterbo.optimization.solvers import (
    LinearProgram,
    ScipyLinearProgramBackend,
    MixedIntegerLinearProgram,
    solve_mixed_integer_linear_program,
    solve_linear_program,
)


def test_linear_program_validation_rejects_mismatched_rhs() -> None:
    objective = jnp.ones(2)
    with pytest.raises(ValueError):
        LinearProgram(objective=objective, lhs_ineq=jnp.ones((1, 2)), rhs_ineq=jnp.ones(2))


def test_scipy_backend_solves_simple_problem() -> None:
    problem = LinearProgram(
        objective=jnp.array([1.0, 0.0]),
        lhs_ineq=jnp.array([[-1.0, 0.0], [0.0, -1.0]]),
        rhs_ineq=jnp.zeros(2),
        lhs_eq=jnp.array([[1.0, 1.0]]),
        rhs_eq=jnp.array([1.0]),
    )
    backend = ScipyLinearProgramBackend()
    solution = backend.solve(problem)
    assert solution.status == "optimal"
    np.testing.assert_allclose(solution.x, np.array([0.0, 1.0]), rtol=1e-9, atol=1e-8)
    assert math.isclose(solution.objective_value, 0.0, rel_tol=0.0, abs_tol=1e-8)


def test_solve_linear_program_uses_default_backend() -> None:
    problem = LinearProgram(
        objective=jnp.array([0.0, 1.0]),
        lhs_ineq=jnp.array([[-1.0, 0.0], [0.0, -1.0]]),
        rhs_ineq=jnp.zeros(2),
        lhs_eq=jnp.array([[1.0, 1.0]]),
        rhs_eq=jnp.array([1.0]),
    )
    solution = solve_linear_program(problem)
    assert solution.status == "optimal"
    np.testing.assert_allclose(solution.x, np.array([1.0, 0.0]), rtol=1e-9, atol=1e-8)


def test_mixed_integer_solver_respects_integrality_constraints() -> None:
    problem = MixedIntegerLinearProgram(
        objective=jnp.array([1.0, 0.1]),
        lhs_ineq=jnp.array([[1.0, 2.0]]),
        rhs_ineq=jnp.array([1.2]),
        lhs_geq=jnp.array([[1.0, 0.0]]),
        rhs_geq=jnp.array([1.0]),
        bounds=[(0.0, None), (0.0, None)],
        integrality=jnp.array([True, False]),
    )

    solution = solve_mixed_integer_linear_program(problem)

    assert solution.status == "Optimal"
    np.testing.assert_allclose(solution.x, np.array([1.0, 0.0]), rtol=1e-9, atol=1e-9)
    assert math.isclose(solution.objective_value, 1.0, rel_tol=0.0, abs_tol=1e-9)


def test_mixed_integer_solver_supports_maximisation() -> None:
    problem = MixedIntegerLinearProgram(
        objective=jnp.array([1.0, 1.0]),
        lhs_ineq=jnp.array([[1.0, 1.0]]),
        rhs_ineq=jnp.array([1.5]),
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        maximize=True,
    )

    solution = solve_mixed_integer_linear_program(problem)

    assert solution.status == "Optimal"
    assert np.all(solution.x >= -1e-9)
    assert np.all(solution.x <= 1.0 + 1e-9)
    assert math.isclose(float(np.sum(solution.x)), 1.5, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(solution.objective_value, 1.5, rel_tol=0.0, abs_tol=1e-9)
