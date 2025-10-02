"""Solver abstractions for linear optimisation problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from jaxtyping import Float

_DIMENSION_AXIS = "dimension"
_INEQUALITY_AXIS = "num_inequalities"
_EQUALITY_AXIS = "num_equalities"

__all__ = [
    "LinearProgram",
    "LinearProgramSolution",
    "LinearProgramBackend",
    "ScipyLinearProgramBackend",
    "CvxpyLinearProgramBackend",
    "solve_linear_program",
]


@dataclass(slots=True)
class LinearProgram:
    """Data structure describing a dense linear program."""

    objective: Float[np.ndarray, _DIMENSION_AXIS]
    lhs_ineq: Float[np.ndarray, f"{_INEQUALITY_AXIS} {_DIMENSION_AXIS}"] | None = None
    rhs_ineq: Float[np.ndarray, _INEQUALITY_AXIS] | None = None
    lhs_eq: Float[np.ndarray, f"{_EQUALITY_AXIS} {_DIMENSION_AXIS}"] | None = None
    rhs_eq: Float[np.ndarray, _EQUALITY_AXIS] | None = None
    bounds: Sequence[tuple[float | None, float | None]] | None = None

    def __post_init__(self) -> None:
        """Normalise array inputs and validate dimension compatibility."""
        objective = np.asarray(self.objective, dtype=float)
        object.__setattr__(self, "objective", objective)

        if objective.ndim != 1:
            msg = "Objective must be a one-dimensional cost vector."
            raise ValueError(msg)

        if self.lhs_ineq is not None:
            lhs = np.asarray(self.lhs_ineq, dtype=float)
            rhs = np.asarray(self.rhs_ineq, dtype=float) if self.rhs_ineq is not None else None
            if lhs.ndim != 2 or lhs.shape[1] != objective.shape[0]:
                msg = "Inequality matrix must have shape (m, n) matching objective dimension."
                raise ValueError(msg)
            if rhs is None or rhs.shape != (lhs.shape[0],):
                msg = "Inequality RHS must have shape (m,)."
                raise ValueError(msg)
            object.__setattr__(self, "lhs_ineq", lhs)
            object.__setattr__(self, "rhs_ineq", rhs)
        elif self.rhs_ineq is not None:
            msg = "Inequality RHS provided without coefficients."
            raise ValueError(msg)

        if self.lhs_eq is not None:
            lhs = np.asarray(self.lhs_eq, dtype=float)
            rhs = np.asarray(self.rhs_eq, dtype=float) if self.rhs_eq is not None else None
            if lhs.ndim != 2 or lhs.shape[1] != objective.shape[0]:
                msg = "Equality matrix must have shape (p, n) matching objective dimension."
                raise ValueError(msg)
            if rhs is None or rhs.shape != (lhs.shape[0],):
                msg = "Equality RHS must have shape (p,)."
                raise ValueError(msg)
            object.__setattr__(self, "lhs_eq", lhs)
            object.__setattr__(self, "rhs_eq", rhs)
        elif self.rhs_eq is not None:
            msg = "Equality RHS provided without coefficients."
            raise ValueError(msg)

        if self.bounds is not None and len(self.bounds) != objective.shape[0]:
            msg = "Bounds length must match the number of variables."
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        """Number of optimisation variables."""
        return int(self.objective.shape[0])


@dataclass(frozen=True, slots=True)
class LinearProgramSolution:
    """Container with the result of a linear program solve."""

    x: Float[np.ndarray, _DIMENSION_AXIS]
    objective_value: float
    status: str


class LinearProgramBackend(Protocol):
    """Protocol implemented by concrete linear program backends."""

    def solve(
        self,
        problem: LinearProgram,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> LinearProgramSolution:
        """Solve ``problem`` and return a :class:`LinearProgramSolution`."""
        ...


class ScipyLinearProgramBackend:
    """Linear program backend using :func:`scipy.optimize.linprog`."""

    def solve(
        self,
        problem: LinearProgram,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> LinearProgramSolution:
        """Solve ``problem`` using :func:`scipy.optimize.linprog`."""
        from scipy.optimize import linprog

        result = linprog(
            c=np.asarray(problem.objective, dtype=float),
            A_ub=None if problem.lhs_ineq is None else np.asarray(problem.lhs_ineq, dtype=float),
            b_ub=None if problem.rhs_ineq is None else np.asarray(problem.rhs_ineq, dtype=float),
            A_eq=None if problem.lhs_eq is None else np.asarray(problem.lhs_eq, dtype=float),
            b_eq=None if problem.rhs_eq is None else np.asarray(problem.rhs_eq, dtype=float),
            bounds=problem.bounds,
            **({} if options is None else dict(options)),
        )

        status = "optimal" if result.success else f"failed({result.status})"
        vector = (
            np.asarray(result.x, dtype=float)
            if result.x is not None
            else np.full(problem.dimension, np.nan)
        )
        objective_value = float(result.fun) if result.fun is not None else float("nan")
        solution = LinearProgramSolution(
            x=vector,
            objective_value=objective_value,
            status=status,
        )
        return solution


class CvxpyLinearProgramBackend:
    """Linear program backend powered by :mod:`cvxpy`."""

    def solve(
        self,
        problem: LinearProgram,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> LinearProgramSolution:
        """Solve ``problem`` using a :mod:`cvxpy` backend."""
        try:
            import cvxpy as cp  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when cvxpy is absent.
            msg = "cvxpy is not installed; install it separately (e.g. `uv pip install cvxpy`)."
            raise ModuleNotFoundError(msg) from exc

        x = cp.Variable(problem.dimension)
        constraints: list[Any] = []

        if problem.lhs_ineq is not None and problem.rhs_ineq is not None:
            constraints.append(cp.matmul(problem.lhs_ineq, x) <= problem.rhs_ineq)

        if problem.lhs_eq is not None and problem.rhs_eq is not None:
            constraints.append(cp.matmul(problem.lhs_eq, x) == problem.rhs_eq)

        if problem.bounds is not None:
            for index, (lower, upper) in enumerate(problem.bounds):
                if lower is not None:
                    constraints.append(x[index] >= lower)
                if upper is not None:
                    constraints.append(x[index] <= upper)

        objective = cp.Minimize(problem.objective @ x)
        optimisation = cp.Problem(objective, constraints)
        optimisation.solve(**({} if options is None else dict(options)))

        if x.value is None:
            msg = f"cvxpy solver failed with status {optimisation.status}."
            raise RuntimeError(msg)

        solution = LinearProgramSolution(
            x=np.asarray(x.value, dtype=float).reshape(problem.dimension),
            objective_value=float(optimisation.value),
            status=str(optimisation.status),
        )
        return solution


def solve_linear_program(
    problem: LinearProgram,
    *,
    backend: LinearProgramBackend | None = None,
    options: Mapping[str, Any] | None = None,
) -> LinearProgramSolution:
    """Solve ``problem`` using ``backend`` or the SciPy fallback."""
    solver = backend or ScipyLinearProgramBackend()
    return solver.solve(problem, options=options)
