"""Reference facet-normal EHZ capacity computation (Google style)."""

from __future__ import annotations

from itertools import permutations
from typing import Final

import numpy as np
from jaxtyping import Float

from viterbo.symplectic.capacity_algorithms._subset_utils import (
    FacetSubset,
    iter_index_combinations,
    prepare_subset,
    subset_capacity_candidate_dynamic,
)
from viterbo.symplectic.core import standard_symplectic_matrix

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_M_AXIS: Final[str] = "m"
_MAX_PERMUTATION_SIZE: Final[int] = 7


def compute_ehz_capacity_reference(
    B_matrix: Float[np.ndarray, f"{_FACET_AXIS} {_DIMENSION_AXIS}"],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    tol: float = 1e-10,
) -> float:
    r"""
    Compute the Ekeland–Hofer–Zehnder capacity of a polytope (reference).

    Args:
      B_matrix: Outward facet normals for ``P = {x : Bx <= c}``, dimension ``d = 2n``.
      c: Offsets ``c``.
      tol: Tolerance for feasibility checks.

    Returns:
      The EHZ capacity of ``P`` under the standard symplectic structure.

    Notes:
      Implements the facet-based optimization formula of Haim–Kislev. For each
      subset of ``2n+1`` facets we solve for Reeb measures and enumerate all
      orders, taking the minimum over admissible subsets.

    Raises:
      ValueError: If ``d`` is not even and >= 2, or no admissible subset passes
        non-negativity constraints.

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

        candidate_value = _subset_capacity_candidate(subset, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not np.isfinite(best_capacity):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return best_capacity


def _subset_capacity_candidate(subset: FacetSubset, *, tol: float) -> float | None:
    beta = subset.beta
    W = subset.symplectic_products
    m = beta.shape[0]
    indices = range(m)

    if m > _MAX_PERMUTATION_SIZE:
        return subset_capacity_candidate_dynamic(subset, tol=tol)

    maximal_value = -np.inf
    for ordering in permutations(indices):
        total = 0.0
        for i in range(1, m):
            idx_i = ordering[i]
            weight_i = beta[idx_i]
            if weight_i <= tol:
                continue
            row = W[idx_i]
            for j in range(i):
                idx_j = ordering[j]
                weight_j = beta[idx_j]
                if weight_j <= tol:
                    continue
                total += weight_i * weight_j * row[idx_j]

        maximal_value = max(maximal_value, total)

    if maximal_value <= tol:
        return None

    return 0.5 / maximal_value
