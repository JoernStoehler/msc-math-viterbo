"""Symplectic capacity algorithms.

Exports reference and fast implementations of EHZ capacity. Prefer importing
specific algorithms for clarity, e.g.:

- ``from viterbo.symplectic.capacity.facet_normals.reference import compute_ehz_capacity_reference``
- ``from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast``
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
