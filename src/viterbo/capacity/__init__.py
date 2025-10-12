"""Modern EHZ capacity heuristics implemented without legacy dependencies."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.capacity import facet_normals, milp, minkowski_billiards, reeb_cycles
from viterbo.capacity import support_relaxation, symmetry_reduced
from viterbo.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
    support_radii,
)
from viterbo.capacity.minkowski_billiards import (
    MinkowskiNormalFan,
    build_normal_fan,
    minkowski_billiard_length_fast,
    minkowski_billiard_length_reference,
)
from viterbo.capacity.reeb_cycles import (
    ehz_capacity_fast_reeb,
    ehz_capacity_reference_reeb,
)
from viterbo.capacity.support_relaxation import (
    support_relaxation_capacity_fast,
    support_relaxation_capacity_reference,
)
from viterbo.capacity.symmetry_reduced import (
    detect_opposite_facet_pairs,
    ehz_capacity_fast_symmetry_reduced,
    ehz_capacity_reference_symmetry_reduced,
)
from viterbo import volume as _volume
from viterbo.numerics import FACET_SOLVER_TOLERANCE
from viterbo.types import FacetPairing, MilpCapacityBounds, SupportRelaxationSummary


def available_solvers() -> tuple[str, ...]:
    """Return solver identifiers supported by the high-level capacity adapters."""

    return tuple(
        sorted(
            {
                "facet-normal-fast",
                "facet-normal-reference",
                "milp-fast",
                "milp-reference",
                "minkowski-fast",
                "minkowski-reference",
                "reeb-fast",
                "reeb-reference",
                "support-fast",
                "support-reference",
                "symmetry-fast",
                "symmetry-reference",
            }
        )
    )


def ehz_capacity_reference(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    vertices: Float[Array, " num_vertices dimension"],
    *,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Reference EHZ capacity obtained via the Haim–Kislev facet solver."""

    normals = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    vertices = jnp.asarray(vertices, dtype=jnp.float64)
    dim = int(normals.shape[1]) if normals.ndim == 2 else 0
    if dim == 2:
        # In 2D, c_EHZ equals area exactly; use area for reference.
        return float(_volume.volume_reference(vertices))
    return float(ehz_capacity_reference_facet_normals(normals, offsets, tol=tol))


def ehz_capacity_fast(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Fast EHZ capacity using the dynamic-programming facet solver."""

    normals = jnp.asarray(normals, dtype=jnp.float64)
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    return float(ehz_capacity_fast_facet_normals(normals, offsets, tol=tol))


ehz_capacity_reference_milp = milp.ehz_capacity_reference_milp
ehz_capacity_fast_milp = milp.ehz_capacity_fast_milp


__all__ = [
    "available_solvers",
    "ehz_capacity_reference",
    "ehz_capacity_fast",
    "FacetPairing",
    "MilpCapacityBounds",
    "SupportRelaxationSummary",
    "MinkowskiNormalFan",
    "detect_opposite_facet_pairs",
    "build_normal_fan",
    "ehz_capacity_fast_facet_normals",
    "ehz_capacity_fast_milp",
    "ehz_capacity_fast_reeb",
    "ehz_capacity_fast_symmetry_reduced",
    "ehz_capacity_reference_facet_normals",
    "ehz_capacity_reference_milp",
    "ehz_capacity_reference_reeb",
    "ehz_capacity_reference_symmetry_reduced",
    "minkowski_billiard_length_fast",
    "minkowski_billiard_length_reference",
    "support_relaxation_capacity_fast",
    "support_relaxation_capacity_reference",
    "support_radii",
    "facet_normals",
    "milp",
    "minkowski_billiards",
    "reeb_cycles",
    "support_relaxation",
    "symmetry_reduced",
]
