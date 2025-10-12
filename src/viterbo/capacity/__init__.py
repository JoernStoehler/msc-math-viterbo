"""Modern EHZ capacity heuristics implemented without legacy dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

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
from viterbo.numerics import FACET_SOLVER_TOLERANCE
from viterbo import volume as _volume
from viterbo.types import Polytope


def available_solvers() -> tuple[str, ...]:
    """Return solver identifiers accepted by :func:`ehz_capacity_batched`."""
    return tuple(
        sorted(
            {
                "facet-normal-reference",
                "facet-normal-fast",
                "reeb-reference",
                "reeb-fast",
                "symmetry-reference",
                "symmetry-fast",
                "support-reference",
                "support-fast",
                "milp-reference",
                "milp-fast",
                "minkowski-reference",
                "minkowski-fast",
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


def _is_valid_bundle(bundle: Polytope) -> bool:
    return bool(bundle.normals.size and bundle.offsets.size)


_SOLVER_DISPATCH: dict[str, Callable[[Polytope], float]] = {
    "facet-normal-reference": facet_normals.ehz_capacity_reference_facet_normals,
    "facet-normal-fast": facet_normals.ehz_capacity_fast_facet_normals,
    "reeb-reference": reeb_cycles.ehz_capacity_reference_reeb,
    "reeb-fast": reeb_cycles.ehz_capacity_fast_reeb,
    "symmetry-reference": symmetry_reduced.ehz_capacity_reference_symmetry_reduced,
    "symmetry-fast": symmetry_reduced.ehz_capacity_fast_symmetry_reduced,
    "support-reference": lambda bundle, **kwargs: support_relaxation.support_relaxation_capacity_reference(
        bundle, **kwargs
    ).capacity_upper_bound,
    "support-fast": lambda bundle, **kwargs: support_relaxation.support_relaxation_capacity_fast(bundle, **kwargs).capacity_upper_bound,
    "milp-reference": lambda bundle, **kwargs: milp.ehz_capacity_reference_milp(bundle, **kwargs).upper_bound,
    "milp-fast": lambda bundle, **kwargs: milp.ehz_capacity_fast_milp(bundle, **kwargs).upper_bound,
}

_MINKOWSKI_DISPATCH: dict[str, Callable[[Polytope, Polytope], float]] = {
    "minkowski-reference": minkowski_billiard_length_reference,
    "minkowski-fast": minkowski_billiard_length_fast,
}

ehz_capacity_reference_milp = milp.ehz_capacity_reference_milp
ehz_capacity_fast_milp = milp.ehz_capacity_fast_milp


def ehz_capacity_batched(
    bundles: Sequence[Polytope],
    *,
    solver: str = "facet-normal-fast",
    geometry: Sequence[Polytope] | Polytope | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> Float[Array, " batch"]:
    """Evaluate ``solver`` on ``bundles`` and return NaN-padded results."""
    if solver not in _SOLVER_DISPATCH and solver not in _MINKOWSKI_DISPATCH:
        msg = f"Unknown solver '{solver}'."
        raise ValueError(msg)

    kwargs = dict(solver_kwargs or {})
    outputs: list[float] = []

    if solver in _MINKOWSKI_DISPATCH:
        solver_fn = _MINKOWSKI_DISPATCH[solver]
        for index, bundle in enumerate(bundles):
            geom_bundle = _geometry_for_index(geometry, index, default=bundle)
            if not (_is_valid_bundle(bundle) and _is_valid_bundle(geom_bundle)):
                outputs.append(float("nan"))
                continue
            try:
                value = solver_fn(bundle, geom_bundle, **kwargs)
                outputs.append(float(value))
            except Exception:  # pragma: no cover
                outputs.append(float("nan"))
        return jnp.asarray(outputs, dtype=jnp.float64)

    solver_fn = _SOLVER_DISPATCH[solver]
    for bundle in bundles:
        if not _is_valid_bundle(bundle):
            outputs.append(float("nan"))
            continue
        try:
            value = solver_fn(bundle, **kwargs)
            outputs.append(float(value))
        except Exception:  # pragma: no cover
            outputs.append(float("nan"))
    return jnp.asarray(outputs, dtype=jnp.float64)


def _geometry_for_index(
    geometry: Sequence[Polytope] | Polytope | None,
    index: int,
    *,
    default: Polytope,
) -> Polytope:
    if geometry is None:
        return default
    if isinstance(geometry, Polytope):
        return geometry
    if len(geometry) <= index:
        msg = "Geometry sequence must match bundle sequence length."
        raise ValueError(msg)
    return geometry[index]


__all__ = [
    "available_solvers",
    "ehz_capacity_reference",
    "ehz_capacity_fast",
    "ehz_capacity_batched",
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
    "facet_normals",
    "milp",
    "minkowski_billiards",
    "reeb_cycles",
    "support_relaxation",
    "symmetry_reduced",
]
