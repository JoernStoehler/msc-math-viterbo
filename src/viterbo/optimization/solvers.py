"""Dense linear program representations and SciPy-backed solvers (wrapped)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypeGuard, cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from viterbo._wrapped.highs import load_highs
from viterbo._wrapped.optimize import linprog as _wrapped_linprog

BoundTuple = tuple[float | None, float | None]


class _BoundsProtocol(Protocol):
    """Structural protocol for bounds-like objects (JAX-first).

    Matches the attributes of ``scipy.optimize.Bounds`` without importing NumPy.
    """

    lb: object | Sequence[float | None] | float | None
    ub: object | Sequence[float | None] | float | None


class _OptimizeResultProtocol(Protocol):
    """Subset of the SciPy ``OptimizeResult`` interface used by linprog."""

    x: object | None
    fun: float | None
    success: bool
    status: int | str
    message: str | None


class _LinprogCallable(Protocol):
    def __call__(
        self,
        *,
        c: Float[Array, " dimension"],
        A_ub: Float[Array, " num_inequalities dimension"] | None,
        b_ub: Float[Array, " num_inequalities"] | None,
        A_eq: Float[Array, " num_equalities dimension"] | None,
        b_eq: Float[Array, " num_equalities"] | None,
        bounds: Sequence[BoundTuple] | None,
        **options: Any,
    ) -> _OptimizeResultProtocol: ...


@lru_cache(1)
def _load_linprog() -> _LinprogCallable:
    """Return the wrapped ``linprog`` callable with a static type signature."""
    return cast(_LinprogCallable, _wrapped_linprog)


@dataclass(frozen=True)
class _HighsResources:
    """Container for HiGHS classes loaded lazily to avoid import costs."""

    Highs: type[Any]
    HighsModelStatus: Any
    HighsStatus: Any
    HighsVarType: Any


@lru_cache(1)
def _load_highs() -> _HighsResources:
    """Load HiGHS classes on-demand."""
    resources = load_highs()

    return _HighsResources(
        Highs=resources.Highs,
        HighsModelStatus=resources.HighsModelStatus,
        HighsStatus=resources.HighsStatus,
        HighsVarType=resources.HighsVarType,
    )


def _is_bounds_object(candidate: object) -> TypeGuard[_BoundsProtocol]:
    """Return ``True`` if ``candidate`` exposes ``lb``/``ub`` arrays."""
    return hasattr(candidate, "lb") and hasattr(candidate, "ub")


def _coerce_bound_value(value: float | None) -> float | None:
    """Convert ``value`` to a finite float or ``None`` for unbounded entries."""
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        msg = "Bounds must not contain NaN entries."
        raise ValueError(msg)
    if not math.isfinite(numeric):
        return None
    return numeric


def _normalize_bounds(
    bounds: Sequence[BoundTuple] | _BoundsProtocol,
    dimension: int,
) -> tuple[BoundTuple, ...]:
    """Validate and canonicalise ``bounds`` for SciPy's ``linprog``."""
    normalized: list[BoundTuple]

    if _is_bounds_object(bounds):
        lower_array = jnp.asarray(bounds.lb, dtype=jnp.float64)
        upper_array = jnp.asarray(bounds.ub, dtype=jnp.float64)

        if lower_array.size == 1:
            lower_array = jnp.full(dimension, float(lower_array.item()))
        if upper_array.size == 1:
            upper_array = jnp.full(dimension, float(upper_array.item()))

        if lower_array.size != dimension or upper_array.size != dimension:
            msg = "Bounds lb/ub must match the number of variables."
            raise ValueError(msg)

        normalized = []
        for idx in range(dimension):
            lower_value = _coerce_bound_value(float(lower_array[idx].item()))
            upper_value = _coerce_bound_value(float(upper_array[idx].item()))
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


@dataclass(slots=True)
class LinearProgram:
    """Data structure describing a dense linear program."""

    objective: Float[Array, " dimension"]
    lhs_ineq: Float[Array, " num_inequalities dimension"] | None = None
    rhs_ineq: Float[Array, " num_inequalities"] | None = None
    lhs_eq: Float[Array, " num_equalities dimension"] | None = None
    rhs_eq: Float[Array, " num_equalities"] | None = None
    bounds: Sequence[BoundTuple] | _BoundsProtocol | None = None

    def __post_init__(self) -> None:
        """Normalise array inputs and validate dimension compatibility."""
        objective = jnp.asarray(self.objective, dtype=jnp.float64)
        object.__setattr__(self, "objective", objective)

        if objective.ndim != 1:
            msg = "Objective must be a one-dimensional cost vector."
            raise ValueError(msg)

        if self.lhs_ineq is not None:
            lhs = jnp.asarray(self.lhs_ineq, dtype=jnp.float64)
            rhs = (
                jnp.asarray(self.rhs_ineq, dtype=jnp.float64) if self.rhs_ineq is not None else None
            )
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
            lhs = jnp.asarray(self.lhs_eq, dtype=jnp.float64)
            rhs = jnp.asarray(self.rhs_eq, dtype=jnp.float64) if self.rhs_eq is not None else None
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

    x: Float[Array, " dimension"]
    objective_value: float
    status: str


def _normalize_integrality_mask(
    integrality: Sequence[bool] | Array | None,
    dimension: int,
) -> tuple[bool, ...]:
    """Return a tuple describing which variables are integral."""
    if integrality is None:
        return tuple(False for _ in range(dimension))

    mask = jnp.asarray(integrality, dtype=jnp.bool_)
    if mask.ndim != 1 or mask.shape[0] != dimension:
        msg = (
            "Integrality mask must be one-dimensional with length equal to the number of variables."
        )
        raise ValueError(msg)
    return tuple(bool(int(value)) for value in mask)


def _coerce_constraint_matrix(
    matrix: Float[Array, " rows dimension"] | None,
    rhs: Float[Array, " rows"] | None,
    dimension: int,
    kind: str,
) -> tuple[Float[Array, " rows dimension"] | None, Float[Array, " rows"] | None]:
    """Validate and coerce constraint matrices to float64."""
    if matrix is None:
        if rhs is not None:
            msg = f"{kind} RHS provided without coefficients."
            raise ValueError(msg)
        return None, None

    lhs = jnp.asarray(matrix, dtype=jnp.float64)
    if lhs.ndim != 2 or lhs.shape[1] != dimension:
        msg = f"{kind} matrix must have shape (m, n) matching objective dimension."
        raise ValueError(msg)

    if rhs is None:
        msg = f"{kind} RHS must be provided when coefficients are supplied."
        raise ValueError(msg)

    rhs_array = jnp.asarray(rhs, dtype=jnp.float64)
    if rhs_array.shape != (lhs.shape[0],):
        msg = f"{kind} RHS must have shape (m,)."
        raise ValueError(msg)

    return lhs, rhs_array


@dataclass(slots=True)
class MixedIntegerLinearProgram:
    """Data structure describing a dense mixed-integer linear program."""

    objective: Float[Array, " dimension"]
    lhs_ineq: Float[Array, " num_ineq dimension"] | None = None
    rhs_ineq: Float[Array, " num_ineq"] | None = None
    lhs_geq: Float[Array, " num_geq dimension"] | None = None
    rhs_geq: Float[Array, " num_geq"] | None = None
    lhs_eq: Float[Array, " num_eq dimension"] | None = None
    rhs_eq: Float[Array, " num_eq"] | None = None
    bounds: Sequence[BoundTuple] | _BoundsProtocol | None = None
    integrality: Sequence[bool] | Array | None = None
    maximize: bool = False

    def __post_init__(self) -> None:
        """Normalise array inputs, integrality mask, and bounds."""
        objective = jnp.asarray(self.objective, dtype=jnp.float64)
        if objective.ndim != 1:
            msg = "Objective must be a one-dimensional cost vector."
            raise ValueError(msg)
        object.__setattr__(self, "objective", objective)

        lhs_leq, rhs_leq = _coerce_constraint_matrix(
            self.lhs_ineq, self.rhs_ineq, objective.shape[0], "Inequality (<=)"
        )
        object.__setattr__(self, "lhs_ineq", lhs_leq)
        object.__setattr__(self, "rhs_ineq", rhs_leq)

        lhs_geq, rhs_geq = _coerce_constraint_matrix(
            self.lhs_geq, self.rhs_geq, objective.shape[0], "Inequality (>=)"
        )
        object.__setattr__(self, "lhs_geq", lhs_geq)
        object.__setattr__(self, "rhs_geq", rhs_geq)

        lhs_eq, rhs_eq = _coerce_constraint_matrix(
            self.lhs_eq, self.rhs_eq, objective.shape[0], "Equality"
        )
        object.__setattr__(self, "lhs_eq", lhs_eq)
        object.__setattr__(self, "rhs_eq", rhs_eq)

        if self.bounds is not None:
            normalized_bounds = _normalize_bounds(self.bounds, objective.shape[0])
            object.__setattr__(self, "bounds", normalized_bounds)

        mask = _normalize_integrality_mask(self.integrality, objective.shape[0])
        object.__setattr__(self, "integrality", mask)
        object.__setattr__(self, "maximize", bool(self.maximize))

    @property
    def dimension(self) -> int:
        """Number of optimisation variables."""
        return int(self.objective.shape[0])


@dataclass(frozen=True, slots=True)
class MixedIntegerLinearProgramSolution:
    """Result container for a mixed-integer solve."""

    x: Float[Array, " dimension"]
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
            c=problem.objective,
            A_ub=None if problem.lhs_ineq is None else problem.lhs_ineq,
            b_ub=None if problem.rhs_ineq is None else problem.rhs_ineq,
            A_eq=None if problem.lhs_eq is None else problem.lhs_eq,
            b_eq=None if problem.rhs_eq is None else problem.rhs_eq,
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
            jnp.asarray(result.x, dtype=jnp.float64)
            if result.x is not None
            else jnp.full(problem.dimension, float("nan"))
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


def _iter_active_constraint_rows(
    matrix: Float[Array, " rows dimension"] | None,
    rhs: Float[Array, " rows"] | None,
) -> Iterable[tuple[np.ndarray, float]]:
    """Yield dense rows with their RHS values for convenience."""
    if matrix is None or rhs is None:
        return []
    matrix_np = np.asarray(matrix, dtype=np.float64)
    rhs_np = np.asarray(rhs, dtype=np.float64)
    return [
        (matrix_np[row_index], float(rhs_np[row_index])) for row_index in range(matrix_np.shape[0])
    ]


def solve_mixed_integer_linear_program(
    problem: MixedIntegerLinearProgram,
    *,
    options: Mapping[str, Any] | None = None,
) -> MixedIntegerLinearProgramSolution:
    """Solve ``problem`` using the HiGHS MILP backend.

    Raises:
        RuntimeError: If HiGHS reports an error or fails to find an optimal solution.
    """

    resources = _load_highs()
    highs = resources.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("random_seed", 0)

    if options is not None:
        for key, value in options.items():
            highs.setOptionValue(str(key), value)

    dimension = problem.dimension
    objective = np.asarray(problem.objective, dtype=np.float64)
    infinity = highs.getInfinity()

    if problem.bounds is None:
        lower_bounds = np.full(dimension, -infinity, dtype=np.float64)
        upper_bounds = np.full(dimension, infinity, dtype=np.float64)
    else:
        bounds_sequence = cast(tuple[BoundTuple, ...], problem.bounds)
        lower_bounds = np.empty(dimension, dtype=np.float64)
        upper_bounds = np.empty(dimension, dtype=np.float64)
        for index, (lower_raw, upper_raw) in enumerate(bounds_sequence):
            lower_bounds[index] = -infinity if lower_raw is None else float(lower_raw)
            upper_bounds[index] = infinity if upper_raw is None else float(upper_raw)

    var_types = [
        resources.HighsVarType.kInteger if is_integer else resources.HighsVarType.kContinuous
        for is_integer in cast(tuple[bool, ...], problem.integrality)
    ]

    variables = highs.addVariables(
        range(dimension),
        lb=lower_bounds.tolist(),
        ub=upper_bounds.tolist(),
        obj=objective.tolist(),
        type=var_types,
    )

    def _add_constraint(row: np.ndarray, rhs_value: float, sense: str) -> None:
        expression = highs.expr()
        for column_index, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            expression += float(coefficient) * variables[column_index]
        if sense == "leq":
            highs.addConstr(expression <= rhs_value)
        elif sense == "geq":
            highs.addConstr(expression >= rhs_value)
        else:
            highs.addConstr(expression == rhs_value)

    for row, rhs_value in _iter_active_constraint_rows(problem.lhs_ineq, problem.rhs_ineq):
        _add_constraint(row, rhs_value, "leq")
    for row, rhs_value in _iter_active_constraint_rows(problem.lhs_geq, problem.rhs_geq):
        _add_constraint(row, rhs_value, "geq")
    for row, rhs_value in _iter_active_constraint_rows(problem.lhs_eq, problem.rhs_eq):
        _add_constraint(row, rhs_value, "eq")

    if problem.maximize:
        run_status = highs.maximize()
    else:
        run_status = highs.minimize()

    if run_status != resources.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS execution failed with status {run_status}")

    model_status = highs.getModelStatus()
    status_string = highs.modelStatusToString(model_status)

    if model_status != resources.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS did not find an optimal solution: {status_string}")

    solution = highs.getSolution()
    vector = jnp.asarray(solution.col_value, dtype=jnp.float64)
    objective_value = float(highs.getObjectiveValue())
    final_status = str(status_string)

    return MixedIntegerLinearProgramSolution(
        x=vector,
        objective_value=objective_value,
        status=final_status,
    )
