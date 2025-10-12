"""Thin wrappers around CVXPy with explicit optional-dependency handling."""

from __future__ import annotations

import cvxpy
from typing import Any

import numpy as _np

_OPTIMAL_STATUSES = frozenset({"optimal", "optimal_inaccurate"})


def solve_epigraph_minimum(
    values: Any,
    *,
    solver: str,
    tolerance: float,
) -> float:
    """Solve ``min t`` s.t. ``t >= values_i`` using CVXPy."""
    variable = cvxpy.Variable()
    constraints: Any = [variable >= float(v) for v in _np.asarray(values, dtype=float)]
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
    # CVXPy lacks precise type hints; treat the solver call as dynamic.
    from typing import cast as _cast, Any as _Any  # local to avoid polluting module namespace

    _cast(_Any, problem).solve(solver=solver, **options)
    status = problem.status.lower()
    if status not in _OPTIMAL_STATUSES:
        msg = f"CVXPy solver failed with status {problem.status}."
        raise RuntimeError(msg)
    assert variable.value is not None
    return float(variable.value)
