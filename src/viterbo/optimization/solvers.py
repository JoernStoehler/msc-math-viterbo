"""Dense linear program representations and SciPy-backed solvers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Protocol, Sequence, TypeGuard, cast

import numpy as np
from jaxtyping import Float

BoundTuple = tuple[float | None, float | None]


class _BoundsProtocol(Protocol):
    """Structural protocol matching ``scipy.optimize.Bounds``."""

    lb: np.ndarray | Sequence[float | None] | float | None
    ub: np.ndarray | Sequence[float | None] | float | None


class _OptimizeResultProtocol(Protocol):
    """Subset of the SciPy ``OptimizeResult`` interface used by linprog."""

    x: np.ndarray | None
    fun: float | None
    success: bool
    status: int | str
    message: str | None


class _LinprogCallable(Protocol):
    def __call__(
        self,
        *,
        c: np.ndarray,
        A_ub: np.ndarray | None,
        b_ub: np.ndarray | None,
        A_eq: np.ndarray | None,
        b_eq: np.ndarray | None,
        bounds: Sequence[BoundTuple] | None,
        **options: Any,
    ) -> _OptimizeResultProtocol: ...


@lru_cache(1)
def _load_linprog() -> _LinprogCallable:
    """Return the SciPy ``linprog`` callable with a static type signature."""
    module = importlib.import_module("scipy.optimize")
    return cast(_LinprogCallable, module.linprog)


def _is_bounds_object(candidate: object) -> TypeGuard[_BoundsProtocol]:
    """Return ``True`` if ``candidate`` exposes ``lb``/``ub`` arrays."""
    return hasattr(candidate, "lb") and hasattr(candidate, "ub")


def _coerce_bound_value(value: float | None) -> float | None:
    """Convert ``value`` to a finite float or ``None`` for unbounded entries."""
    if value is None:
        return None
    numeric = float(value)
    if np.isnan(numeric):
        msg = "Bounds must not contain NaN entries."
        raise ValueError(msg)
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalize_bounds(
    bounds: Sequence[BoundTuple] | _BoundsProtocol,
    dimension: int,
) -> tuple[BoundTuple, ...]:
    """Validate and canonicalise ``bounds`` for SciPy's ``linprog``."""
    normalized: list[BoundTuple]

    if _is_bounds_object(bounds):
        lower_array = np.atleast_1d(np.asarray(bounds.lb, dtype=float))
        upper_array = np.atleast_1d(np.asarray(bounds.ub, dtype=float))

        if lower_array.size == 1:
            lower_array = np.full(dimension, lower_array.item(), dtype=float)
        if upper_array.size == 1:
            upper_array = np.full(dimension, upper_array.item(), dtype=float)

        if lower_array.size != dimension or upper_array.size != dimension:
            msg = "Bounds lb/ub must match the number of variables."
            raise ValueError(msg)

        normalized = []
        for lower, upper in zip(lower_array, upper_array, strict=True):
            lower_value = _coerce_bound_value(float(lower))
            upper_value = _coerce_bound_value(float(upper))
            if lower_value is not None and upper_value is not None and lower_value > upper_value:
                msg = "Lower bound exceeds upper bound."
                raise ValueError(msg)
            normalized.append((lower_value, upper_value))
        return tuple(normalized)

    if not isinstance(bounds, Sequence):
        msg = "Bounds must be a sequence of (lower, upper) pairs."
        raise TypeError(msg)

    sequence_bounds = tuple(bounds)
    if len(sequence_bounds) != dimension:
        msg = "Bounds length must match the number of variables."
        raise ValueError(msg)

    normalized = []
    for index, pair in enumerate(sequence_bounds):
        if len(pair) != 2:
            msg = "Each bounds entry must be a (lower, upper) pair."
            raise ValueError(msg)

        lower_raw, upper_raw = pair
        lower_value = _coerce_bound_value(lower_raw if lower_raw is None else float(lower_raw))
        upper_value = _coerce_bound_value(upper_raw if upper_raw is None else float(upper_raw))

        if lower_value is not None and upper_value is not None and lower_value > upper_value:
            msg = f"Lower bound exceeds upper bound at index {index}."
            raise ValueError(msg)
        normalized.append((lower_value, upper_value))

    return tuple(normalized)


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
    bounds: Sequence[BoundTuple] | _BoundsProtocol | None = None

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

        if self.bounds is not None:
            normalized_bounds = _normalize_bounds(self.bounds, objective.shape[0])
            object.__setattr__(self, "bounds", normalized_bounds)

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
        """Solve ``problem`` using :func:`scipy.optimize.linprog`.

        Raises:
            RuntimeError: If the SciPy solver reports a non-success status.
        """
        linprog_callable = _load_linprog()
        bounds_argument = cast(tuple[BoundTuple, ...] | None, problem.bounds)
        result: _OptimizeResultProtocol = linprog_callable(
            c=np.asarray(problem.objective, dtype=float),
            A_ub=None if problem.lhs_ineq is None else np.asarray(problem.lhs_ineq, dtype=float),
            b_ub=None if problem.rhs_ineq is None else np.asarray(problem.rhs_ineq, dtype=float),
            A_eq=None if problem.lhs_eq is None else np.asarray(problem.lhs_eq, dtype=float),
            b_eq=None if problem.rhs_eq is None else np.asarray(problem.rhs_eq, dtype=float),
            bounds=bounds_argument,
            **({} if options is None else dict(options)),
        )

        if not result.success:
            status_code = getattr(result, "status", "unknown")
            message = getattr(result, "message", "")
            msg = (
                "Linear program solve failed with status"
                f" {status_code!r}: {message or 'no diagnostic message provided.'}"
            )
            raise RuntimeError(msg)

        status = "optimal"
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


def solve_linear_program(
    problem: LinearProgram,
    *,
    backend: LinearProgramBackend | None = None,
    options: Mapping[str, Any] | None = None,
) -> LinearProgramSolution:
    """Solve ``problem`` using ``backend`` or the SciPy fallback."""
    solver = backend or ScipyLinearProgramBackend()
    return solver.solve(problem, options=options)
