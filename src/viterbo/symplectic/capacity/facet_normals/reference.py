"""Reference facet-normal EHZ capacity computation (JAX-first)."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING, overload

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.subset_utils import (
    FacetSubset,
    iter_index_combinations,
    prepare_subset,
    subset_capacity_candidate_dynamic,
)
from viterbo.symplectic.core import standard_symplectic_matrix

if TYPE_CHECKING:  # type-only import to keep runtime free of NumPy
    import numpy as np

_MAX_PERMUTATION_SIZE: int = 7


@overload
def compute_ehz_capacity_reference(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float = 1e-10,
) -> float: ...


@overload
def compute_ehz_capacity_reference(
    B_matrix: Float["np.ndarray", " num_facets dimension"],
    c: Float["np.ndarray", " num_facets"],
    *,
    tol: float = 1e-10,
) -> float: ...


def compute_ehz_capacity_reference(
    B_matrix: Float[Array, " num_facets dimension"] | Float[np.ndarray, " num_facets dimension"],
    c: Float[Array, " num_facets"] | Float[np.ndarray, " num_facets"],
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
    B = jnp.asarray(B_matrix, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)

    if B.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")

    if c.ndim != 1 or c.shape[0] != B.shape[0]:
        raise ValueError("Vector c must have length equal to the number of facets.")

    num_facets, dimension = B.shape
    if int(dimension) % 2 != 0 or int(dimension) < 2:
        raise ValueError("The ambient dimension must satisfy 2n with n >= 1.")

    J = standard_symplectic_matrix(dimension)
    subset_size = dimension + 1
    best_capacity = jnp.inf

    for indices in iter_index_combinations(num_facets, subset_size):
        subset = prepare_subset(B_matrix=B, c=c, indices=indices, J=J, tol=tol)
        if subset is None:
            continue

        candidate_value = _subset_capacity_candidate(subset, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not bool(jnp.isfinite(best_capacity)):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return float(best_capacity)


def _subset_capacity_candidate(subset: FacetSubset, *, tol: float) -> float | None:
    beta = jnp.asarray(subset.beta, dtype=jnp.float64)
    W = jnp.asarray(subset.symplectic_products, dtype=jnp.float64)
    m = int(beta.shape[0])
    indices = range(m)

    if m > _MAX_PERMUTATION_SIZE:
        return subset_capacity_candidate_dynamic(subset, tol=tol)

    maximal_value = float("-inf")
    for ordering in permutations(indices):
        total = 0.0
        for i in range(1, m):
            idx_i = ordering[i]
            weight_i = float(beta[idx_i])
            if weight_i <= float(tol):
                continue
            row = W[idx_i]
            for j in range(i):
                idx_j = ordering[j]
                weight_j = float(beta[idx_j])
                if weight_j <= float(tol):
                    continue
                total += weight_i * weight_j * float(row[idx_j])

        maximal_value = max(maximal_value, total)

    if maximal_value <= float(tol):
        return None

    return 0.5 / maximal_value
