"""Compatibility wrapper re-exporting from `viterbo.math.numerics`."""

from __future__ import annotations

from viterbo.math.numerics import (
    FACET_SOLVER_TOLERANCE,
    GEOMETRY_ABS_TOLERANCE,
    INCIDENCE_ABS_TOLERANCE,
    INCIDENCE_REL_TOLERANCE,
)

__all__ = [
    "GEOMETRY_ABS_TOLERANCE",
    "FACET_SOLVER_TOLERANCE",
    "INCIDENCE_REL_TOLERANCE",
    "INCIDENCE_ABS_TOLERANCE",
]
