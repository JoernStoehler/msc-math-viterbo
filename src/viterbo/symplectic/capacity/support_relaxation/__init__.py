"""Support-function relaxations for EHZ capacity upper bounds."""

from __future__ import annotations

from viterbo.symplectic.capacity.support_relaxation.fast import (
    SupportRelaxationDiagnostics,
    SupportRelaxationResult,
    compute_support_relaxation_capacity_fast,
)
from viterbo.symplectic.capacity.support_relaxation.reference import (
    compute_support_relaxation_capacity_reference,
)

__all__ = [
    "compute_support_relaxation_capacity_fast",
    "compute_support_relaxation_capacity_reference",
    "SupportRelaxationDiagnostics",
    "SupportRelaxationResult",
]
