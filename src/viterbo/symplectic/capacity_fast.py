"""Optimised computation of the Ekeland–Hofer–Zehnder capacity."""

from .capacity_algorithms.facet_normals_fast import compute_ehz_capacity_fast

__all__ = ["compute_ehz_capacity_fast"]
