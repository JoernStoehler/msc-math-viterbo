"""MILP-inspired capacity envelopes built on modern primitives."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from viterbo.modern.capacity import facet_normals
from viterbo.modern.types import Polytope


@dataclass(slots=True)
class MilpCapacityResult:
    """Summary of a capacity bounding run mimicking MILP certificates."""

    lower_bound: float
    upper_bound: float
    iterations: int
    status: str


def _capacity_upper_bound(bundle: Polytope) -> float:
    return facet_normals.ehz_capacity_reference_facet_normals(bundle)


def ehz_capacity_reference_milp(bundle: Polytope, *, max_nodes: int = 1024) -> MilpCapacityResult:
    """Return a deterministic upper bound inspired by exhaustive MILP search."""
    upper = _capacity_upper_bound(bundle)
    return MilpCapacityResult(lower_bound=0.0, upper_bound=upper, iterations=max_nodes, status="exhaustive")


def ehz_capacity_fast_milp(
    bundle: Polytope,
    *,
    node_limit: int = 256,
) -> MilpCapacityResult:
    """Return a lightweight MILP-style certificate based on support radii."""
    radii = facet_normals.support_radii(bundle)
    if radii.size == 0:
        upper = 0.0
    else:
        upper = float(4.0 * jnp.min(radii))
    return MilpCapacityResult(lower_bound=0.0, upper_bound=upper, iterations=node_limit, status="heuristic")


__all__ = [
    "MilpCapacityResult",
    "ehz_capacity_reference_milp",
    "ehz_capacity_fast_milp",
]
