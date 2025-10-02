"""Reference facet-normal EHZ capacity computation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable

import numpy as np
from jaxtyping import Float

from ..core import standard_symplectic_matrix


@dataclass(frozen=True)
class FacetSubset:
    """Data container describing a subset of polytope facets."""

    indices: tuple[int, ...]
    beta: Float[np.ndarray, " m"]  # shape: (m,)
    symplectic_products: Float[np.ndarray, "m m"]  # shape: (m, m)


def compute_ehz_capacity_reference(
    B: Float[np.ndarray, "num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    *,
    tol: float = 1e-10,
) -> float:
    r"""
    Compute the Ekeland–Hofer–Zehnder capacity of a non-degenerate polytope.

    Parameters
    ----------
    B:
        Matrix whose rows describe outward pointing facet normals of a convex
        polytope ``P`` via the inequality description ``P = \{x : B x \leq c\}``.
        The ambient dimension equals ``d = 2n`` with ``n \geq 1``.
    c:
        Vector of offsets appearing in the facet inequalities. The i-th
        inequality is ``\langle b_i, x \rangle \leq c_i``.
    tol:
        Numerical tolerance for feasibility checks. Default ``1e-10``.

    Returns
    -------
    float
        The Ekeland–Hofer–Zehnder capacity of ``P`` under the standard
        symplectic structure.

    Notes
    -----
    The computation implements the facet-based optimisation formula of
    Haim-Kislev (``arXiv:1905.04769``). For each subset of ``2n+1`` facets we
    solve the affine system describing Reeb measures. If the resulting
    weights are non-negative, we evaluate the skew-symmetric quadratic form
    from Haim-Kislev's theorem across every total order of the chosen facets
    (Chaidez–Hutchings provide the bridge between Reeb measures and facet
    orders on polytopes). The reciprocal of the maximal form value—scaled by
    ``1/2``—gives a candidate action; the minimum over all admissible subsets
    equals ``c_{\mathrm{EHZ}}``.

    The algorithm is exponential both in the number of facets (due to the
    subset enumeration) and within each subset (due to permutation
    enumeration). This mirrors the inherent complexity established by
    Leipold–Vallentin (2024) and prioritises correctness over performance.

    Raises
    ------
    ValueError
        If the dimension is not even and at least two or if no admissible
        facet subset satisfies the non-negativity constraints implied by the
        Reeb measure relations.

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
    B: Float[np.ndarray, "num_facets dimension"],
    c: Float[np.ndarray, " num_facets"],
    indices: Iterable[int],
    J: np.ndarray,
    tol: float,
) -> FacetSubset | None:
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

    if not np.allclose(B_subset.T @ beta, np.zeros(B_subset.shape[1]), atol=1e-8):
        return None

    if not np.isclose(float(c_subset @ beta), 1.0, atol=1e-8):
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
