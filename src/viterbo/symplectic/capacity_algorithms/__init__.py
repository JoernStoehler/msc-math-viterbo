"""Facet-normal algorithms used by the EHZ capacity implementations."""

from .facet_normals_fast import compute_ehz_capacity_fast
from .facet_normals_reference import compute_ehz_capacity_reference

__all__ = [
    "compute_ehz_capacity_fast",
    "compute_ehz_capacity_reference",
]
