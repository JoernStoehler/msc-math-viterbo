"""Unit tests for the solver abstraction layer."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from viterbo.optimization.solvers import (
    LinearProgram,
    MixedIntegerLinearProgram,
    ScipyLinearProgramBackend,
    _normalize_bounds,
    solve_linear_program,
    solve_mixed_integer_linear_program,
)


@pytest.mark.goal_code
def test_linear_program_validation_rejects_mismatched_rhs() -> None:
    """LinearProgram rejects RHS vectors whose length mismatches the constraints."""
    objective = jnp.ones(2)
    with pytest.raises(ValueError):
        LinearProgram(objective=objective, lhs_ineq=jnp.ones((1, 2)), rhs_ineq=jnp.ones(2))


@pytest.mark.goal_math
def test_scipy_backend_solves_simple_problem() -> None:
    """The SciPy backend recovers the analytic solution of a toy LP."""
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


@pytest.mark.goal_code
def test_solve_linear_program_uses_default_backend() -> None:
    """`solve_linear_program` delegates to the default backend for the toy LP."""
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


@pytest.mark.goal_math
def test_mixed_integer_solver_respects_integrality_constraints() -> None:
    """The mixed-integer solver honours integrality flags on variables."""
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


@pytest.mark.goal_math
def test_mixed_integer_solver_supports_maximisation() -> None:
    """The solver handles maximisation problems by mirroring the objective."""
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

@pytest.mark.goal_code
def test_normalize_bounds_expands_scalars() -> None:
    """`_normalize_bounds` broadens scalar bounds into per-variable tuples."""
    class DummyBounds:
        def __init__(self) -> None:
            self.lb = -1.0
            self.ub = jnp.array([2.0, jnp.inf, 0.5])

    normalized = _normalize_bounds(DummyBounds(), dimension=3)
    assert normalized == ((-1.0, 2.0), (-1.0, None), (-1.0, 0.5))

@pytest.mark.goal_code
def test_normalize_bounds_validates_sequence() -> None:
    """`_normalize_bounds` enforces length and ordering constraints for sequences."""
    with pytest.raises(ValueError, match="Bounds length must match"):
        _normalize_bounds([(0.0, 1.0)], dimension=2)

    with pytest.raises(ValueError, match="Lower bound exceeds upper bound"):
        _normalize_bounds([(0.0, -0.5)], dimension=1)

@pytest.mark.goal_code
def test_normalize_bounds_rejects_nan_entries() -> None:
    """`_normalize_bounds` refuses NaN inputs to prevent invalid solver requests."""
    with pytest.raises(ValueError, match="Bounds must not contain NaN"):
        _normalize_bounds([(float('nan'), 1.0)], dimension=1)
