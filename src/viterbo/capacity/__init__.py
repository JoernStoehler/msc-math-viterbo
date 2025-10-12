"""Modern EHZ capacity heuristics implemented without legacy dependencies."""

from __future__ import annotations

import jax.numpy as jnp

from viterbo.capacity import facet_normals, milp, minkowski_billiards, reeb_cycles
from viterbo.capacity import support_relaxation, symmetry_reduced
from viterbo.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
    support_radii,
)
from viterbo.capacity.milp import MilpCapacityResult
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
    SupportRelaxationDiagnostics,
    SupportRelaxationResult,
    support_relaxation_capacity_fast,
    support_relaxation_capacity_reference,
)
from viterbo.capacity.symmetry_reduced import (
    FacetPairingMetadata,
    detect_opposite_facet_pairs,
    ehz_capacity_fast_symmetry_reduced,
    ehz_capacity_reference_symmetry_reduced,
)
from viterbo import volume as _volume
from viterbo.numerics import FACET_SOLVER_TOLERANCE
from viterbo.types import Polytope


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
    bundle: Polytope,
    *,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Reference EHZ capacity obtained via the Haim–Kislev facet solver."""

    dim = int(jnp.asarray(bundle.normals, dtype=jnp.float64).shape[1])
    if dim == 2:
        # In 2D, c_EHZ equals area exactly; use area for reference.
        return float(_volume.volume_reference(bundle))
    return float(ehz_capacity_reference_facet_normals(bundle, tol=tol))


def ehz_capacity_fast(
    bundle: Polytope,
    *,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Fast EHZ capacity using the dynamic-programming facet solver."""

    return float(ehz_capacity_fast_facet_normals(bundle, tol=tol))


ehz_capacity_reference_milp = milp.ehz_capacity_reference_milp
ehz_capacity_fast_milp = milp.ehz_capacity_fast_milp


__all__ = [
    "available_solvers",
    "ehz_capacity_reference",
    "ehz_capacity_fast",
    "FacetPairingMetadata",
    "MilpCapacityResult",
    "SupportRelaxationDiagnostics",
    "SupportRelaxationResult",
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
