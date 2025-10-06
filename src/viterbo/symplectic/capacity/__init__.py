"""Symplectic capacity algorithms.

Exports reference and fast implementations of EHZ capacity, including
symmetry-reduced variants. Prefer importing specific algorithms for clarity,
e.g.:

- ``from viterbo.symplectic.capacity.facet_normals.reference import compute_ehz_capacity_reference``
- ``from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast``
- ``from viterbo.symplectic.capacity.symmetry_reduced import compute_ehz_capacity_fast_symmetry_reduced``
"""

from __future__ import annotations

from viterbo.symplectic.capacity.reeb_cycles.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_reeb_fast,
)
from viterbo.symplectic.capacity.reeb_cycles.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reeb_reference,
)
from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast,
)
from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference,
)
from viterbo.symplectic.capacity.symmetry_reduced import (
    FacetPairingMetadata as FacetPairingMetadata,
    compute_ehz_capacity_fast_symmetry_reduced as compute_ehz_capacity_fast_symmetry_reduced,
    compute_ehz_capacity_reference_symmetry_reduced as compute_ehz_capacity_reference_symmetry_reduced,
    detect_opposite_facet_pairs as detect_opposite_facet_pairs,
)
