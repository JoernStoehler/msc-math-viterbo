"""Optimized facet-normal EHZ capacity computation (Google style)."""

from __future__ import annotations

from typing import Final

import numpy as np
from jaxtyping import Float

from viterbo.symplectic.capacity_algorithms._subset_utils import (
    iter_index_combinations,
    prepare_subset,
    subset_capacity_candidate_dynamic,
)
from viterbo.symplectic.core import standard_symplectic_matrix

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"


def compute_ehz_capacity_fast(
    B_matrix: Float[np.ndarray, f"{_FACET_AXIS} {_DIMENSION_AXIS}"],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    tol: float = 1e-10,
) -> float:
    r"""
    Compute EHZ capacity via a dynamic-programming search over facet orders.

    Mirrors the reference subset enumeration but replaces the inner permutation
    search by a DP that exploits antisymmetry of the Haim–Kislev form, reducing
    ``m!`` to ``O(m^2 2^m)`` for ``m = 2n + 1``.

    Args:
      B_matrix: Outward facet normals for ``P = {x : Bx <= c}``, dimension ``2n``.
      c: Facet offsets ``c``.
      tol: Tolerance for feasibility and zero weights.

    Returns:
      The EHZ capacity under the standard symplectic form.

    Raises:
      ValueError: If no admissible facet subset satisfies Reeb-measure constraints.

    """
    B = np.asarray(B_matrix, dtype=float)
    c = np.asarray(c, dtype=float)

    if B.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")

    if c.ndim != 1 or c.shape[0] != B.shape[0]:
        raise ValueError("Vector c must have length equal to the number of facets.")

    num_facets, dimension = B.shape
    if dimension % 2 != 0 or dimension < 2:
        raise ValueError("The ambient dimension must satisfy 2n with n >= 1.")

    J = standard_symplectic_matrix(dimension)
    subset_size = dimension + 1
    best_capacity = np.inf

    for indices in iter_index_combinations(num_facets, subset_size):
        subset = prepare_subset(B_matrix=B, c=c, indices=indices, J=J, tol=tol)
        if subset is None:
            continue

        candidate_value = subset_capacity_candidate_dynamic(subset, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not np.isfinite(best_capacity):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return best_capacity
