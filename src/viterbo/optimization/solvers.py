"""Solver abstractions for linear optimisation problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import scipy.optimize as _opt  # type: ignore[reportMissingTypeStubs]  # SciPy lacks type stubs; TODO: add stubs or pin typeshed
from jaxtyping import Float

linprog = _opt.linprog  # type: ignore[reportUnknownMemberType]  # Treat as dynamic; SciPy stubs are incomplete. TODO: type this via stub

_DIMENSION_AXIS = "dimension"
_INEQUALITY_AXIS = "num_inequalities"
_EQUALITY_AXIS = "num_equalities"


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
        result: Any = linprog(
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


## Golden-path only: we intentionally omit alternative MILP backends to avoid optional stacks.
## Golden-path only: we intentionally omit alternative MILP backends to avoid optional stacks.


def solve_linear_program(
    problem: LinearProgram,
    *,
    backend: LinearProgramBackend | None = None,
    options: Mapping[str, Any] | None = None,
) -> LinearProgramSolution:
    """Solve ``problem`` using ``backend`` or the SciPy fallback."""
    solver = backend or ScipyLinearProgramBackend()
    return solver.solve(problem, options=options)
