"""Symmetry-reduced capacity heuristics built from modern data structures."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from viterbo.capacity import facet_normals
from viterbo.types import Polytope


@dataclass(slots=True)
class FacetPairingMetadata:
    """Pairs of opposite facets detected from outward normals."""

    pairs: tuple[tuple[int, int], ...]
    unpaired: tuple[int, ...]


def detect_opposite_facet_pairs(
    bundle: Polytope,
    *,
    angle_tolerance: float = 1e-6,
) -> FacetPairingMetadata:
    """Detect opposite facet pairs via cosine similarity."""
    normals = jnp.asarray(bundle.normals, dtype=jnp.float64)
    norms = jnp.linalg.norm(normals, axis=1)
    safe = jnp.where(norms == 0.0, 1.0, norms)
    unit = normals / safe[:, None]
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i in range(unit.shape[0]):
        if i in used:
            continue
        best_index = None
        best_cosine = 1.0
        for j in range(i + 1, unit.shape[0]):
            if j in used:
                continue
            cosine = float(jnp.dot(unit[i], unit[j]))
            if abs(cosine + 1.0) < abs(best_cosine + 1.0):
                best_cosine = cosine
                best_index = j
        if best_index is not None and abs(best_cosine + 1.0) <= angle_tolerance:
            used.add(i)
            used.add(best_index)
            pairs.append((i, best_index))
    unpaired = tuple(idx for idx in range(unit.shape[0]) if idx not in used)
    return FacetPairingMetadata(pairs=tuple(pairs), unpaired=unpaired)


def _reduced_radii(bundle: Polytope, pairing: FacetPairingMetadata | None) -> jnp.ndarray:
    radii = facet_normals.support_radii(bundle)
    if radii.size == 0:
        return radii
    if pairing is None:
        return radii
    paired_values = []
    for left, right in pairing.pairs:
        paired_values.append(0.5 * (radii[left] + radii[right]))
    if pairing.unpaired:
        unpaired_values = radii[jnp.array(pairing.unpaired, dtype=jnp.int32)]
        all_values = jnp.concatenate((jnp.asarray(paired_values, dtype=jnp.float64), unpaired_values))
    else:
        all_values = jnp.asarray(paired_values, dtype=jnp.float64)
    if all_values.size == 0:
        return radii
    return all_values


def ehz_capacity_reference_symmetry_reduced(
    bundle: Polytope,
    *,
    pairing: FacetPairingMetadata | None = None,
) -> float:
    """Reference symmetry-reduced capacity delegating to the facet solver."""

    try:
        return float(facet_normals.ehz_capacity_reference_facet_normals(bundle))
    except ValueError:
        effective_radii = _reduced_radii(bundle, pairing or detect_opposite_facet_pairs(bundle))
        if effective_radii.size == 0:
            return 0.0
        return float(4.0 * jnp.min(effective_radii))


def ehz_capacity_fast_symmetry_reduced(
    bundle: Polytope,
    *,
    pairing: FacetPairingMetadata | None = None,
) -> float:
    """Fast symmetry-reduced capacity identical to the reference variant."""
    return ehz_capacity_reference_symmetry_reduced(bundle, pairing=pairing)


__all__ = [
    "FacetPairingMetadata",
    "detect_opposite_facet_pairs",
    "ehz_capacity_reference_symmetry_reduced",
    "ehz_capacity_fast_symmetry_reduced",
]
