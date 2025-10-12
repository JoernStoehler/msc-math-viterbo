"""Facet-normal EHZ capacity solvers for the modern namespace.

This module ports the full Haim–Kislev subset machinery into the flat
``viterbo`` namespace while keeping the original helper exposed for
callers that only need support radii. The reference solver enumerates
facet subsets exactly, whereas the fast variant applies the dynamic
programming shortcut used in the legacy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterator, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic import standard_symplectic_matrix
from viterbo.types import Polytope
from viterbo.numerics import FACET_SOLVER_TOLERANCE

np = jnp

_MAX_PERMUTATION_SIZE = 7


@dataclass(frozen=True)
class _FacetSubset:
    """Data describing a facet subset and its Reeb measure."""

    indices: tuple[int, ...]
    beta: Float[Array, " m"]
    symplectic_products: Float[Array, " m m"]


def support_radii(bundle: Polytope) -> Float[Array, " num_facets"]:
    """Return the radial support values ``offset / ||normal||`` for each facet."""

    normals = jnp.asarray(bundle.normals, dtype=jnp.float64)
    offsets = jnp.asarray(bundle.offsets, dtype=jnp.float64)
    if normals.ndim != 2 or offsets.ndim != 1:
        return jnp.asarray([], dtype=jnp.float64)
    if normals.shape[0] == 0:
        return jnp.asarray([], dtype=jnp.float64)
    norms = jnp.linalg.norm(normals, axis=1)
    safe_norms = jnp.where(norms == 0.0, 1.0, norms)
    radii = offsets / safe_norms
    return jnp.clip(radii, a_min=0.0)


def _bundle_arrays(
    bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    if isinstance(bundle, Polytope):
        normals = jnp.asarray(bundle.normals, dtype=jnp.float64)
        offsets = jnp.asarray(bundle.offsets, dtype=jnp.float64)
    else:
        normals, offsets = bundle
        normals = jnp.asarray(normals, dtype=jnp.float64)
        offsets = jnp.asarray(offsets, dtype=jnp.float64)
    return normals, offsets


def _iter_index_combinations(count: int, size: int) -> Iterator[tuple[int, ...]]:
    for combo in combinations(range(count), size):
        yield tuple(int(index) for index in combo)


def _prepare_subset(
    *,
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    indices: Sequence[int],
    J: Float[Array, " dimension dimension"],
    tol: float,
) -> _FacetSubset | None:
    selected = tuple(int(index) for index in indices)
    row_indices = jnp.asarray(selected, dtype=jnp.int64)
    B_subset = jnp.take(B_matrix, row_indices, axis=0)
    c_subset = jnp.take(c, row_indices, axis=0)
    m = int(B_subset.shape[0])

    system = jnp.zeros((m, m), dtype=jnp.float64)
    system = system.at[0, :].set(c_subset)
    system = system.at[1:, :].set(B_subset.T)

    rhs = jnp.zeros((m,), dtype=jnp.float64)
    rhs = rhs.at[0].set(1.0)

    try:
        beta = jnp.linalg.solve(system, rhs)
    except (TypeError, ValueError, RuntimeError):
        return None

    beta = jnp.where(jnp.abs(beta) <= float(tol), 0.0, beta)
    if bool(jnp.any(beta < -float(tol))):
        return None

    if not bool(np.allclose(B_subset.T @ beta, jnp.zeros(B_subset.shape[1]), atol=float(tol), rtol=0.0)):
        return None

    if not bool(np.isclose(float(c_subset @ beta), 1.0, atol=float(tol), rtol=0.0)):
        return None

    symplectic_products = (B_subset @ jnp.asarray(J)) @ B_subset.T
    return _FacetSubset(indices=selected, beta=beta, symplectic_products=symplectic_products)


def _maximum_antisymmetric_order_value(weights: Array) -> float:
    w = jnp.asarray(weights, dtype=jnp.float64)
    m = int(w.shape[0])
    if m == 0:
        return 0.0

    size = 1 << m
    dp = jnp.full(size, -jnp.inf)
    dp = dp.at[0].set(0.0)

    prefix_sums = jnp.zeros((m, size), dtype=jnp.float64)
    for idx in range(m):
        for mask in range(1, size):
            lsb = mask & -mask
            bit_index = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            prefix_sums = prefix_sums.at[idx, mask].set(prefix_sums[idx, prev_mask] + float(w[idx, bit_index]))

    for mask in range(size):
        current = float(dp[mask])
        if not bool(jnp.isfinite(current)):
            continue

        remaining = (~mask) & (size - 1)
        while remaining:
            lsb = remaining & -remaining
            next_index = lsb.bit_length() - 1
            new_mask = mask | lsb
            candidate = current + float(prefix_sums[next_index, mask])
            if candidate > float(dp[new_mask]):
                dp = dp.at[new_mask].set(candidate)
            remaining ^= lsb

    return float(dp[-1])


def _subset_capacity_candidate_dynamic(subset: _FacetSubset, *, tol: float) -> float | None:
    beta = jnp.asarray(subset.beta, dtype=jnp.float64)
    W = jnp.asarray(subset.symplectic_products, dtype=jnp.float64)

    positive = jnp.where(beta > float(tol))[0]
    if int(positive.size) < 2:
        return None

    beta_active = beta[positive]
    W_active = W[jnp.ix_(positive, positive)]
    weights = jnp.multiply(jnp.multiply.outer(beta_active, beta_active), W_active)
    weights = weights - jnp.diag(jnp.diag(weights))

    maximal_value = _maximum_antisymmetric_order_value(weights)
    if maximal_value <= float(tol):
        return None

    return 0.5 / maximal_value


def _subset_capacity_candidate(subset: _FacetSubset, *, tol: float) -> float | None:
    beta = jnp.asarray(subset.beta, dtype=jnp.float64)
    W = jnp.asarray(subset.symplectic_products, dtype=jnp.float64)
    m = int(beta.shape[0])
    indices = range(m)

    if m > _MAX_PERMUTATION_SIZE:
        return _subset_capacity_candidate_dynamic(subset, tol=tol)

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


def _compute_ehz_capacity_reference(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float,
) -> float:
    if B_matrix.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")
    if c.ndim != 1 or c.shape[0] != B_matrix.shape[0]:
        raise ValueError("Vector c must match the number of facets.")

    num_facets, dimension = B_matrix.shape
    if dimension == 0 or num_facets == 0:
        return 0.0
    if int(dimension) % 2 != 0 or int(dimension) < 2:
        raise ValueError("The ambient dimension must satisfy 2n with n >= 1.")

    J = standard_symplectic_matrix(int(dimension))
    subset_size = int(dimension) + 1
    best_capacity = jnp.inf

    for indices in _iter_index_combinations(num_facets, subset_size):
        subset = _prepare_subset(B_matrix=B_matrix, c=c, indices=indices, J=J, tol=tol)
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


def _compute_ehz_capacity_fast(
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    tol: float,
) -> float:
    if B_matrix.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")
    if c.ndim != 1 or c.shape[0] != B_matrix.shape[0]:
        raise ValueError("Vector c must match the number of facets.")

    num_facets, dimension = B_matrix.shape
    if dimension == 0 or num_facets == 0:
        return 0.0
    if int(dimension) % 2 != 0 or int(dimension) < 2:
        raise ValueError("The ambient dimension must satisfy 2n with n >= 1.")

    J = standard_symplectic_matrix(int(dimension))
    subset_size = int(dimension) + 1
    best_capacity = jnp.inf

    for indices in _iter_index_combinations(num_facets, subset_size):
        subset = _prepare_subset(B_matrix=B_matrix, c=c, indices=indices, J=J, tol=tol)
        if subset is None:
            continue
        candidate_value = _subset_capacity_candidate_dynamic(subset, tol=tol)
        if candidate_value is None:
            continue
        if candidate_value < best_capacity:
            best_capacity = candidate_value

    if not bool(jnp.isfinite(best_capacity)):
        raise ValueError("No admissible facet subset satisfied the non-negativity constraints.")

    return float(best_capacity)


def ehz_capacity_reference_facet_normals(
    bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]],
    *,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Reference Haim–Kislev solver for the EHZ capacity."""

    B_matrix, offsets = _bundle_arrays(bundle)
    if B_matrix.size == 0 or offsets.size == 0:
        return 0.0
    return _compute_ehz_capacity_reference(B_matrix, offsets, tol=tol)


def ehz_capacity_fast_facet_normals(
    bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]],
    *,
    tol: float = FACET_SOLVER_TOLERANCE,
) -> float:
    """Dynamic-programming shortcut mirroring the legacy fast solver."""

    B_matrix, offsets = _bundle_arrays(bundle)
    if B_matrix.size == 0 or offsets.size == 0:
        return 0.0
    return _compute_ehz_capacity_fast(B_matrix, offsets, tol=tol)


__all__ = [
    "support_radii",
    "ehz_capacity_reference_facet_normals",
    "ehz_capacity_fast_facet_normals",
]
