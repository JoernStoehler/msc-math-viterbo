"""Shared helpers for facet-subset preparation across EHZ solvers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
from jaxtyping import Float

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_M_AXIS: Final[str] = "m"


@dataclass(frozen=True)
class FacetSubset:
    """Data describing a subset of polytope facets."""

    indices: tuple[int, ...]
    beta: Float[np.ndarray, f"{_M_AXIS}"]  # shape: (m,)
    symplectic_products: Float[np.ndarray, f"{_M_AXIS} {_M_AXIS}"]  # shape: (m, m)


def iter_index_combinations(count: int, size: int) -> Iterator[tuple[int, ...]]:
    """Yield index tuples of length ``size`` drawn from ``count`` facets."""

    for combination in combinations(range(count), size):
        yield tuple(int(index) for index in combination)


def prepare_subset(
    *,
    B_matrix: Float[np.ndarray, f"{_FACET_AXIS} {_DIMENSION_AXIS}"],
    c: Float[np.ndarray, _FACET_AXIS],
    indices: Iterable[int],
    J: np.ndarray,
    tol: float,
) -> FacetSubset | None:
    """Solve the Reeb-measure system for a facet subset and cache products."""

    selected_tuple = tuple(int(index) for index in indices)
    row_indices = np.array(selected_tuple, dtype=int)
    B_subset = B_matrix[row_indices, :]
    c_subset = c[row_indices]
    m = B_subset.shape[0]

    system = np.zeros((m, m))
    system[0, :] = c_subset
    system[1:, :] = B_subset.T

    rhs = np.zeros(m)
    rhs[0] = 1.0

    try:
        beta = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None

    beta[np.abs(beta) <= tol] = 0.0
    if np.any(beta < -tol):
        return None

    if not np.allclose(
        B_subset.T @ beta,
        np.zeros(B_subset.shape[1]),
        atol=tol,
        rtol=0.0,
    ):
        return None

    if not np.isclose(float(c_subset @ beta), 1.0, atol=tol, rtol=0.0):
        return None

    symplectic_products = (B_subset @ J) @ B_subset.T
    return FacetSubset(indices=selected_tuple, beta=beta, symplectic_products=symplectic_products)


def maximum_antisymmetric_order_value(weights: np.ndarray) -> float:
    r"""Return the maximum order value for an antisymmetric weight matrix."""

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

    return float(dp[-1])


def subset_capacity_candidate_dynamic(
    subset: FacetSubset,
    *,
    tol: float,
) -> float | None:
    """Return the candidate capacity using the dynamic-programming shortcut."""

    beta = subset.beta
    symplectic_products = subset.symplectic_products

    positive = np.where(beta > tol)[0]
    if positive.size < 2:
        return None

    beta_active = beta[positive]
    W_active = symplectic_products[np.ix_(positive, positive)]
    weights = np.multiply.outer(beta_active, beta_active) * W_active

    np.fill_diagonal(weights, 0.0)

    maximal_value = maximum_antisymmetric_order_value(weights)
    if maximal_value <= tol:
        return None

    return 0.5 / maximal_value


__all__ = [
    "FacetSubset",
    "iter_index_combinations",
    "maximum_antisymmetric_order_value",
    "prepare_subset",
    "subset_capacity_candidate_dynamic",
]
