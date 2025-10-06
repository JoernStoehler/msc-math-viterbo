"""Utilities for detecting opposite-facet pairs and encoding symmetry metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from jaxtyping import Array, Float


@dataclass(frozen=True)
class FacetPairingMetadata:
    """Symmetry metadata describing opposite facets of a polytope."""

    pairs: tuple[tuple[int, int], ...]
    unpaired: tuple[int, ...]

    def __post_init__(self) -> None:
        """Normalise pairs/unpaired facets and validate disjointness."""
        seen: set[int] = set()
        normalised_pairs: list[tuple[int, int]] = []
        for first, second in self.pairs:
            if first == second:
                raise ValueError("Facet pairs must reference two distinct facets.")
            a, b = int(first), int(second)
            ordered_pair: tuple[int, int] = (a, b) if a <= b else (b, a)
            if ordered_pair[0] in seen or ordered_pair[1] in seen:
                raise ValueError("Facet indices cannot appear in more than one pair.")
            seen.update(ordered_pair)
            normalised_pairs.append(ordered_pair)
        object.__setattr__(self, "pairs", tuple(normalised_pairs))

        unpaired_indices = tuple(int(idx) for idx in self.unpaired)
        for idx in unpaired_indices:
            if idx in seen:
                raise ValueError("Unpaired indices overlap with paired facets.")
        object.__setattr__(self, "unpaired", unpaired_indices)

    @property
    def partner_lookup(self) -> dict[int, int]:
        """Return a lookup mapping each facet to its opposite partner."""
        lookup: dict[int, int] = {}
        for first, second in self.pairs:
            lookup[first] = second
            lookup[second] = first
        return lookup

    def is_canonical_subset(self, indices: Sequence[int]) -> bool:
        """Return ``True`` if ``indices`` respects canonical representatives."""
        ordered = [int(i) for i in indices]
        positions: dict[int, int] = {}
        for position, index in enumerate(ordered):
            if index not in positions:
                positions[index] = position

        for first, second in self.pairs:
            first_present = first in positions
            second_present = second in positions
            if second_present and not first_present:
                return False
            if first_present and second_present and positions[second] < positions[first]:
                return False
        return True

    def subset_groups(self, indices: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Return tuples of facet indices constrained to share ``β`` weights."""
        partners = self.partner_lookup
        subset_order = [int(i) for i in indices]
        present = set(subset_order)
        groups: list[tuple[int, ...]] = []
        visited: set[int] = set()
        for index in subset_order:
            if index in visited:
                continue
            partner = partners.get(index)
            if partner is not None and partner in present:
                group = tuple(sorted((index, partner)))
                visited.update(group)
            else:
                group = (index,)
                visited.add(index)
            groups.append(group)
        return tuple(groups)

    def orbit_representatives(self) -> tuple[int, ...]:
        """Return canonical orbit representatives (pairs' minima and unpaired facets)."""
        return tuple(sorted({pair[0] for pair in self.pairs} | set(self.unpaired)))


def detect_opposite_facet_pairs(
    B_matrix: (Float[Array, " num_facets dimension"] | np.ndarray | Iterable[Sequence[float]]),
    c: Float[Array, " num_facets"] | np.ndarray | Sequence[float],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    overrides: Sequence[tuple[int, int]] | None = None,
) -> FacetPairingMetadata:
    """Detect opposite-facet pairs via normals and offsets.

    The detector assumes facets are provided as ``P = {x : B x <= c}`` and flags two
    facets ``i`` and ``j`` as opposite when ``B[i]`` and ``B[j]`` are negatives of
    each other (within ``atol``/``rtol`` tolerances) and ``c[i]`` equals ``c[j]``.
    This heuristic matches centrally symmetric polytopes whose symmetries act by
    negating coordinates. Users can override the detected pairing by passing
    ``overrides`` with explicit index pairs, in which case all remaining facets are
    treated as unpaired. Override indices must be disjoint.
    """
    B = np.asarray(B_matrix, dtype=np.float64)
    offsets = np.asarray(c, dtype=np.float64)
    num_facets = int(B.shape[0])

    if overrides is not None:
        provided_pairs = tuple((int(i0), int(i1)) for (i0, i1) in overrides)
        used = {idx for pair in provided_pairs for idx in pair}
        if any(idx < 0 or idx >= num_facets for idx in used):
            raise ValueError("Override indices must lie within the facet range.")
        unpaired = tuple(idx for idx in range(num_facets) if idx not in used)
        return FacetPairingMetadata(pairs=provided_pairs, unpaired=unpaired)

    remaining = set(range(num_facets))
    detected_pairs: list[tuple[int, int]] = []
    while remaining:
        i = min(remaining)
        remaining.remove(i)
        normal = B[i]
        offset = offsets[i]
        partner: int | None = None
        for j in list(remaining):
            if np.isclose(offsets[j], offset, atol=atol, rtol=rtol) and np.allclose(
                B[j], -normal, atol=atol, rtol=rtol
            ):
                partner = j
                break
        if partner is not None:
            remaining.remove(partner)
            detected_pairs.append((i, partner))
        else:
            continue
    paired_indices = {idx for pair in detected_pairs for idx in pair}
    unpaired = tuple(idx for idx in range(num_facets) if idx not in paired_indices)
    return FacetPairingMetadata(pairs=tuple(detected_pairs), unpaired=unpaired)
