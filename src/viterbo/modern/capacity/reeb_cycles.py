"""Piecewise-linear Reeb cycle heuristics for modern bundles."""

from __future__ import annotations

import jax.numpy as jnp

from viterbo.modern.capacity import facet_normals
from viterbo.modern.types import Polytope


def ehz_capacity_reference_reeb(bundle: Polytope) -> float:
    """Reference cycle-based estimate equal to the facet-normal heuristic."""
    radii = facet_normals.support_radii(bundle)
    if radii.size == 0:
        return 0.0
    return float(4.0 * jnp.min(radii))


def ehz_capacity_fast_reeb(bundle: Polytope) -> float:
    """Fast Reeb-cycle heuristic identical to the reference variant."""
    return ehz_capacity_reference_reeb(bundle)


__all__ = [
    "ehz_capacity_reference_reeb",
    "ehz_capacity_fast_reeb",
]
