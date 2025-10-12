"""Shared numerical constants for the modern namespace (math layer)."""

from __future__ import annotations

GEOMETRY_ABS_TOLERANCE: float = 1e-9
"""Default absolute tolerance for combinatorial geometry routines."""

FACET_SOLVER_TOLERANCE: float = 1e-10
"""Default feasibility tolerance for facet-based capacity solvers."""

INCIDENCE_REL_TOLERANCE: float = 1e-12
"""Default relative tolerance when checking vertex-facet incidence."""

INCIDENCE_ABS_TOLERANCE: float = 0.0
"""Default absolute tolerance when checking vertex-facet incidence."""
