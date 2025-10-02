"""Optimized facet-normal EHZ capacity computation (Google style)."""

from __future__ import annotations

from itertools import combinations
from typing import Final

import numpy as np
from jaxtyping import Float

from ..core import standard_symplectic_matrix
from .facet_normals_reference import FacetSubset, _prepare_subset

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"


def compute_ehz_capacity_fast(
    B: Float[np.ndarray, f"{_FACET_AXIS} {_DIMENSION_AXIS}"],
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
      B: Outward facet normals for ``P = {x : Bx <= c}``, dimension ``2n``.
      c: Facet offsets ``c``.
      tol: Tolerance for feasibility and zero weights.

    Returns:
      The EHZ capacity under the standard symplectic form.

    Raises:
      ValueError: If no admissible facet subset satisfies Reeb-measure constraints.

    """
    B = np.asarray(B, dtype=float)
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

    for indices in combinations(range(num_facets), subset_size):
        subset = _prepare_subset(B=B, c=c, indices=indices, J=J, tol=tol)
        if subset is None:
            continue

        candidate_value = _subset_capacity_candidate_fast(subset, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not np.isfinite(best_capacity):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return best_capacity


def _subset_capacity_candidate_fast(subset: FacetSubset, *, tol: float) -> float | None:
    beta = subset.beta
    symplectic_products = subset.symplectic_products

    positive = np.where(beta > tol)[0]
    if positive.size < 2:
        return None

    beta_active = beta[positive]
    W_active = symplectic_products[np.ix_(positive, positive)]
    weights = np.multiply.outer(beta_active, beta_active) * W_active

    np.fill_diagonal(weights, 0.0)

    maximal_value = _maximum_antisymmetric_order_value(weights)
    if maximal_value <= tol:
        return None

    return 0.5 / maximal_value


def _maximum_antisymmetric_order_value(weights: np.ndarray) -> float:
    r"""
    Return maximum order value for an antisymmetric weight matrix.

    Args:
      weights: Square matrix ``W`` with ``W[i, j] = -W[j, i]`` after weighting
        by Reeb measures.

    Returns:
      Maximal order value obtained by summing row prefixes.

    """
    m = weights.shape[0]
    if m == 0:
        return 0.0

    size = 1 << m
    dp = np.full(size, -np.inf)
    dp[0] = 0.0

    prefix_sums = np.zeros((m, size))
    for idx in range(m):
        for mask in range(1, size):
            lsb = mask & -mask
            bit_index = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            prefix_sums[idx, mask] = prefix_sums[idx, prev_mask] + weights[idx, bit_index]

    for mask in range(size):
        current = dp[mask]
        if not np.isfinite(current):
            continue

        remaining = (~mask) & (size - 1)
        while remaining:
            lsb = remaining & -remaining
            next_index = lsb.bit_length() - 1
            new_mask = mask | lsb
            candidate = current + prefix_sums[next_index, mask]
            if candidate > dp[new_mask]:
                dp[new_mask] = candidate
            remaining ^= lsb

    return dp[-1]


__all__ = ["compute_ehz_capacity_fast"]
