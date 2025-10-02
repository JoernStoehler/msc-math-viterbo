"""Optimised facet-normal EHZ capacity computation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from jaxtyping import Float

from ..core import standard_symplectic_matrix
from .facet_normals_reference import FacetSubset, _prepare_subset


def compute_ehz_capacity_fast(
    B: Float[np.ndarray, "num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    tol: float = 1e-10,
) -> float:
    r"""
    Compute the EHZ capacity using a dynamic-programming search over facet orders.

    The outer structure of the algorithm mirrors
    :func:`viterbo.symplectic.capacity_algorithms.facet_normals_reference.compute_ehz_capacity_reference`
    but replaces the exhaustive permutation search within each facet subset by a
    dynamic program. The DP exploits the antisymmetry of the Haim–Kislev
    bilinear form to reduce the ``m!`` search over orders of ``m = 2n + 1`` facets
    to ``O(m^2 2^m)`` states. For the small values of ``m`` relevant to symplectic
    polytopes (``m <= 9`` in our experiments) this yields a pronounced speed-up
    while staying in pure NumPy.

    Parameters
    ----------
    B:
        Outward pointing facet normals describing the polytope ``P = {x : Bx <= c}``.
        The dimension of the ambient space equals ``2n`` with ``n >= 1``.
    c:
        Facet offsets for the inequality representation.
    tol:
        Numerical tolerance for feasibility checks and zero weights.

    Returns
    -------
    float
        The EHZ capacity of the polytope under the standard symplectic form.

    Raises
    ------
    ValueError
        If no admissible facet subset satisfies the non-negativity constraints in
        the Reeb measure relations.

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
    Return the maximum order value for an antisymmetric weight matrix.

    Parameters
    ----------
    weights:
        Square matrix ``W`` with ``W[i, j] = -W[j, i]`` encoding the bilinear form
        contributions between active facets after weighting by the Reeb measure.

    Returns
    -------
    float
        The maximal order value obtained by summing row prefixes.

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
