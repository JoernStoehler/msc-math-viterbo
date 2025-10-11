"""Modern EHZ capacity heuristics implemented without legacy dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.modern.capacity import facet_normals, milp, minkowski_billiards, reeb_cycles
from viterbo.modern.capacity import support_relaxation, symmetry_reduced
from viterbo.modern.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
    support_radii,
)
from viterbo.modern.capacity.milp import MilpCapacityResult
from viterbo.modern.capacity.minkowski_billiards import (
    minkowski_billiard_length_fast,
    minkowski_billiard_length_reference,
)
from viterbo.modern.capacity.reeb_cycles import (
    ehz_capacity_fast_reeb,
    ehz_capacity_reference_reeb,
)
from viterbo.modern.capacity.support_relaxation import (
    SupportRelaxationDiagnostics,
    SupportRelaxationResult,
    support_relaxation_capacity_fast,
    support_relaxation_capacity_reference,
)
from viterbo.modern.capacity.symmetry_reduced import (
    FacetPairingMetadata,
    detect_opposite_facet_pairs,
    ehz_capacity_fast_symmetry_reduced,
    ehz_capacity_reference_symmetry_reduced,
)
from viterbo.modern.types import Polytope


def _polygon_area(vertices: Float[Array, " num_vertices 2"]) -> float:
    if vertices.shape[0] < 3:
        return 0.0
    centroid = jnp.mean(vertices, axis=0)
    rel = vertices - centroid
    angles = jnp.arctan2(rel[:, 1], rel[:, 0])
    order = jnp.argsort(angles)
    ordered = vertices[order]
    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - y * jnp.roll(x, -1)))
    return float(area)


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


def ehz_capacity_reference(bundle: Polytope) -> float:
    """Reference EHZ capacity heuristic."""
    dimension = int(bundle.vertices.shape[1]) if bundle.vertices.ndim else 0
    if dimension == 2 and bundle.vertices.size:
        return _polygon_area(bundle.vertices)
    radii = support_radii(bundle)
    if radii.size == 0:
        return 0.0
    return float(4.0 * jnp.min(radii))


def ehz_capacity_fast(bundle: Polytope, *, tol: float = 1e-10) -> float:
    """Fast EHZ capacity heuristic matching the facet-normal estimator."""
    return float(ehz_capacity_fast_facet_normals(bundle, tol=tol))


def _is_valid_bundle(bundle: Polytope) -> bool:
    return bool(bundle.normals.size and bundle.offsets.size and bundle.vertices.size)


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
    "detect_opposite_facet_pairs",
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
