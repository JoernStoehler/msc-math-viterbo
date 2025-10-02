"""Reference computation of the Ekeland–Hofer–Zehnder capacity."""

from .capacity_algorithms.facet_normals_reference import compute_ehz_capacity_reference

compute_ehz_capacity = compute_ehz_capacity_reference

__all__ = ["compute_ehz_capacity"]
