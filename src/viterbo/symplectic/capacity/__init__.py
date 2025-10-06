"""Symplectic capacity algorithms.

Exports reference and fast implementations of EHZ capacity. Prefer importing
specific algorithms for clarity, e.g.:

- ``from viterbo.symplectic.capacity.facet_normals.reference import compute_ehz_capacity_reference``
- ``from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast``
"""

from __future__ import annotations

from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast,
)
from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference,
)
from viterbo.symplectic.capacity.minkowski_billiards import (
    MinkowskiNormalFan as MinkowskiNormalFan,
    build_normal_fan as build_normal_fan,
    compute_minkowski_billiard_length_fast as compute_minkowski_billiard_length_fast,
    compute_minkowski_billiard_length_reference as compute_minkowski_billiard_length_reference,
)
