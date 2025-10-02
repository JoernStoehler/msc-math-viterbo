"""Backward-compatible wrapper for the reference facet-normal algorithm."""

from viterbo.algorithms.facet_normals_reference import compute_ehz_capacity_reference

compute_ehz_capacity = compute_ehz_capacity_reference

__all__ = ["compute_ehz_capacity"]
