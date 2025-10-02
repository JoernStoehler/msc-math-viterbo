"""Reference facet-normal EHZ capacity computation (Google style)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Final, Iterable

import numpy as np
from jaxtyping import Float

from ..core import standard_symplectic_matrix

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"


@dataclass(frozen=True)
class FacetSubset:
    """Data describing a subset of polytope facets."""

    indices: tuple[int, ...]
    beta: Float[np.ndarray, " m"]  # shape: (m,)
    symplectic_products: Float[np.ndarray, "m m"]  # shape: (m, m)


def compute_ehz_capacity_reference(
    B: Float[np.ndarray, f"{_FACET_AXIS} {_DIMENSION_AXIS}"],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    tol: float = 1e-10,
) -> float:
    r"""
    Compute the Ekeland–Hofer–Zehnder capacity of a polytope (reference).

    Args:
      B: Outward facet normals for ``P = {x : Bx <= c}``, dimension ``d = 2n``.
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

        candidate_value = _subset_capacity_candidate(subset, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not np.isfinite(best_capacity):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return best_capacity


def _prepare_subset(
    *,
    B: Float[np.ndarray, f"{_FACET_AXIS} {_DIMENSION_AXIS}"],
    c: Float[np.ndarray, _FACET_AXIS],
    indices: Iterable[int],
    J: np.ndarray,
    tol: float,
) -> FacetSubset | None:
    """
    Solve Reeb-measure system for a facet subset and build cached data.

    Args:
      B: Facet normals.
      c: Offsets.
      indices: Chosen facet indices.
      J: Symplectic matrix.
      tol: Numerical tolerance.

    Returns:
      ``FacetSubset`` if feasible and non-negative, otherwise ``None``.

    """
    selected_tuple = tuple(indices)
    row_indices = np.array(selected_tuple, dtype=int)
    B_subset = B[row_indices, :]
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


def _subset_capacity_candidate(subset: FacetSubset, *, tol: float) -> float | None:
    beta = subset.beta
    W = subset.symplectic_products
    m = beta.shape[0]
    indices = range(m)

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


__all__ = [
    "FacetSubset",
    "compute_ehz_capacity_reference",
]
