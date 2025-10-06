"""Thin wrappers around CVXPy with solver tolerance helpers."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as _np

_OPTIMAL_STATUSES = frozenset({"optimal", "optimal_inaccurate"})


def _load_cvxpy() -> Any:
    try:
        return importlib.import_module("cvxpy")
    except ModuleNotFoundError as error:  # pragma: no cover - import guard
        msg = (
            "cvxpy is required for the reference support-relaxation solver. "
            "Install the optional dependency or select the fast solver."
        )
        raise ModuleNotFoundError(msg) from error


def solve_epigraph_minimum(
    values: Any,
    *,
    solver: str,
    tolerance: float,
) -> float:
    """Solve ``min t`` subject to ``t >= values_i`` using CVXPy."""
    array = _np.asarray(values, dtype=float)
    if array.size == 0:
        msg = "Epigraph minimum expects at least one value."
        raise ValueError(msg)
    cvxpy = _load_cvxpy()

    variable = cvxpy.Variable()
    constraints = [variable >= float(v) for v in array]
    problem = cvxpy.Problem(cvxpy.Minimize(variable), constraints)
    options: dict[str, Any] = {}
    solver_lower = solver.lower()
    if tolerance > 0.0:
        if solver_lower == "scs":
            options["eps_abs"] = tolerance
            options["eps_rel"] = tolerance
        elif solver_lower == "ecos":
            options["abstol"] = tolerance
            options["reltol"] = tolerance
    problem.solve(solver=solver, **options)
    status = problem.status.lower()
    if status not in _OPTIMAL_STATUSES:
        msg = f"CVXPy solver failed with status {problem.status}."
        raise RuntimeError(msg)
    return float(variable.value)
