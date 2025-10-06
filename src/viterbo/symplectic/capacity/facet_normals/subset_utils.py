"""Shared helpers for facet-subset preparation across EHZ solvers (JAX-first)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations

import jax.numpy as jnp
from jaxtyping import Array, Float

# Compatibility alias for tests that monkeypatch subset_utils.np.*
np = jnp


@dataclass(frozen=True)
class FacetSubset:
    """Data describing a subset of polytope facets."""

    indices: tuple[int, ...]
    beta: Float[Array, " m"]  # shape: (m,)
    symplectic_products: Float[Array, " m m"]  # shape: (m, m)


def iter_index_combinations(count: int, size: int) -> Iterator[tuple[int, ...]]:
    """Yield index tuples of length ``size`` drawn from ``count`` facets."""
    for combination in combinations(range(count), size):
        yield tuple(int(index) for index in combination)


def prepare_subset(
    *,
    B_matrix: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    indices: Iterable[int],
    J: Float[Array, " dimension dimension"],
    tol: float,
) -> FacetSubset | None:
    """Solve the Reeb-measure system for a facet subset and cache products (JAX)."""
    selected_tuple = tuple(int(index) for index in indices)
    row_indices = jnp.asarray(selected_tuple, dtype=int)
    B_subset = jnp.asarray(B_matrix)[row_indices, :]
    c_subset = jnp.asarray(c)[row_indices]
    m = int(B_subset.shape[0])

    system = jnp.zeros((m, m), dtype=jnp.float64)
    system = system.at[0, :].set(c_subset)
    system = system.at[1:, :].set(B_subset.T)

    rhs = jnp.zeros(m, dtype=jnp.float64)
    rhs = rhs.at[0].set(1.0)

    # Solve linear system; failures will raise a LinAlgError-like exception, which
    # we conservatively treat as infeasible by returning None.
    try:
        beta = jnp.linalg.solve(system, rhs)
    except Exception:
        return None

    beta = jnp.where(jnp.abs(beta) <= float(tol), 0.0, beta)
    if bool(jnp.any(beta < -float(tol))):
        return None

    if not bool(
        np.allclose(B_subset.T @ beta, jnp.zeros(B_subset.shape[1]), atol=float(tol), rtol=0.0)
    ):
        return None

    if not bool(np.isclose(float(c_subset @ beta), 1.0, atol=float(tol), rtol=0.0)):
        return None

    symplectic_products = (B_subset @ jnp.asarray(J)) @ B_subset.T
    return FacetSubset(indices=selected_tuple, beta=beta, symplectic_products=symplectic_products)


def maximum_antisymmetric_order_value(weights: Array) -> float:
    r"""Return the maximum order value for an antisymmetric weight matrix."""
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
            prefix_sums = prefix_sums.at[idx, mask].set(
                prefix_sums[idx, prev_mask] + float(w[idx, bit_index])
            )

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


def subset_capacity_candidate_dynamic(
    subset: FacetSubset,
    *,
    tol: float,
) -> float | None:
    """Return the candidate capacity using the dynamic-programming shortcut (JAX)."""
    beta = jnp.asarray(subset.beta, dtype=jnp.float64)
    W = jnp.asarray(subset.symplectic_products, dtype=jnp.float64)

    positive = jnp.where(beta > float(tol))[0]
    if int(positive.size) < 2:
        return None

    beta_active = beta[positive]
    W_active = W[jnp.ix_(positive, positive)]
    weights = jnp.multiply(jnp.multiply.outer(beta_active, beta_active), W_active)

    # Zero the diagonal
    weights = weights - jnp.diag(jnp.diag(weights))

    maximal_value = maximum_antisymmetric_order_value(weights)
    if maximal_value <= float(tol):
        return None

    return 0.5 / maximal_value
