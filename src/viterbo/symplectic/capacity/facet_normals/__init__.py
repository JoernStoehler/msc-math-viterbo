"""Facet-normal based algorithms for the EHZ capacity."""

from __future__ import annotations

from viterbo.symplectic.capacity.facet_normals.fast import (
    compute_ehz_capacity_fast as compute_ehz_capacity_fast,
)
from viterbo.symplectic.capacity.facet_normals.reference import (
    compute_ehz_capacity_reference as compute_ehz_capacity_reference,
)
