"""Symplectic capacity algorithms.

Exports reference and fast implementations of EHZ capacity, including
symmetry-reduced variants. Prefer importing specific algorithms for clarity,
e.g.:

- ``from viterbo.symplectic.capacity.facet_normals.reference import compute_ehz_capacity_reference``
- ``from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast``
- ``from viterbo.symplectic.capacity.symmetry_reduced import compute_ehz_capacity_fast_symmetry_reduced``
"""

from __future__ import annotations

from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast
from viterbo.symplectic.capacity.facet_normals.reference import compute_ehz_capacity_reference
from viterbo.symplectic.capacity.milp.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast_milp,
from viterbo.symplectic.capacity.reeb_cycles.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_reeb_fast,
)
from viterbo.symplectic.capacity.reeb_cycles.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reeb_reference,
)
from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast,
)
from viterbo.symplectic.capacity.milp.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference_milp,
)
from viterbo.symplectic.capacity.minkowski_billiards import (
    MinkowskiNormalFan as MinkowskiNormalFan,
    build_normal_fan as build_normal_fan,
    compute_minkowski_billiard_length_fast as compute_minkowski_billiard_length_fast,
    compute_minkowski_billiard_length_reference as compute_minkowski_billiard_length_reference,
from viterbo.symplectic.capacity.support_relaxation.fast import (
    compute_support_relaxation_capacity_fast as compute_support_relaxation_capacity_fast,
)
from viterbo.symplectic.capacity.support_relaxation.reference import (
    compute_support_relaxation_capacity_reference as compute_support_relaxation_capacity_reference,
from viterbo.symplectic.capacity.symmetry_reduced import (
    FacetPairingMetadata as FacetPairingMetadata,
    compute_ehz_capacity_fast_symmetry_reduced as compute_ehz_capacity_fast_symmetry_reduced,
    compute_ehz_capacity_reference_symmetry_reduced as compute_ehz_capacity_reference_symmetry_reduced,
    detect_opposite_facet_pairs as detect_opposite_facet_pairs,
)
