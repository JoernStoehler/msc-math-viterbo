"""Minkowski billiard length heuristics for modern polytopes."""

from __future__ import annotations

import jax.numpy as jnp

from viterbo.modern.capacity import facet_normals
from viterbo.modern.types import Polytope


def _min_radius(bundle: Polytope) -> float:
    radii = facet_normals.support_radii(bundle)
    if radii.size == 0:
        return 0.0
    return float(jnp.min(radii))


def minkowski_billiard_length_reference(table: Polytope, geometry: Polytope) -> float:
    """Reference billiard length combining table and geometry radii."""
    radius_table = _min_radius(table)
    radius_geometry = _min_radius(geometry)
    return float(4.0 * (radius_table + radius_geometry))


def minkowski_billiard_length_fast(table: Polytope, geometry: Polytope) -> float:
    """Fast billiard length heuristic mirroring the reference implementation."""
    return minkowski_billiard_length_reference(table, geometry)


__all__ = [
    "minkowski_billiard_length_reference",
    "minkowski_billiard_length_fast",
]
