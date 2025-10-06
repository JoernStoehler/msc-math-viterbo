"""Optimized facet-normal EHZ capacity computation (JAX-first)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.facet_normals.subset_utils import (
    iter_index_combinations,
    prepare_subset,
    subset_capacity_candidate_dynamic,
)
from viterbo.symplectic.core import standard_symplectic_matrix


def compute_ehz_capacity_fast(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
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

        candidate_value = subset_capacity_candidate_dynamic(subset, tol=tol)
        if candidate_value is None:
            continue

        if candidate_value < float(best_capacity):
            best_capacity = float(candidate_value)

    if not bool(jnp.isfinite(best_capacity)):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return float(best_capacity)
