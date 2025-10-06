"""Fast half-space helpers (speed-optimized, JAX-backed where useful)."""

from __future__ import annotations

import itertools
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.halfspaces import _shared


def _all_index_combinations(count: int, dimension: int) -> jnp.ndarray:
    combos = list(itertools.combinations(range(count), dimension))
    if not combos:
        return jnp.zeros((0, max(dimension, 1)), dtype=jnp.int32)
    return jnp.asarray(combos, dtype=jnp.int32)


@jax.jit
def _feasible_solutions(
    matrix: jnp.ndarray, offsets: jnp.ndarray, indices: jnp.ndarray, atol: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # matrix: [F, D], offsets: [F], indices: [C, D]
    subsets = matrix[indices, :]  # [C, D, D]
    targets = offsets[indices]  # [C, D]

    def solve_one(A: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        s = cast(jnp.ndarray, jnp.linalg.svd(A, compute_uv=False))  # type: ignore[reportUnnecessaryCast]
        smax = jnp.max(s)
        tol = smax * 1e-12
        rank = jnp.sum(s > tol)
        full = rank == A.shape[0]

        def _solve_branch(args: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            mat, vec = args
            return jnp.linalg.solve(mat, vec)

        def _zeros_branch(args: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            return jnp.zeros_like(args[1])

        x = jax.lax.cond(full, _solve_branch, _zeros_branch, (A, b))
        return x, full

    xs, full_mask = jax.vmap(solve_one)(subsets, targets)
    atol_j = jnp.asarray(atol, dtype=offsets.dtype)
    feas = jnp.all(matrix @ xs.T <= offsets[:, None] + atol_j, axis=0)
    mask = jnp.logical_and(full_mask, feas)
    return xs, mask


def _unique_rows_jax(points: jnp.ndarray, *, atol: float) -> jnp.ndarray:
    if points.size == 0:
        return points
    # Sort rows lexicographically for stable dedup with tolerance.
    order = jnp.lexsort(points.T)
    sorted_pts = points[order]

    def step(carry: jnp.ndarray, current: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        last = carry
        atol_j = jnp.asarray(atol, dtype=points.dtype)
        keep = jnp.any(jnp.abs(current - last) > atol_j)
        new_last = jax.lax.select(keep, current, last)
        return new_last, keep

    init = sorted_pts[0]
    _, keep_rest = jax.lax.scan(step, init, sorted_pts[1:])
    keep_mask = jnp.concatenate((jnp.array([True]), keep_rest))
    return jnp.compress(keep_mask, sorted_pts, axis=0)


def enumerate_vertices(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[Array, " num_vertices dimension"]:
    """Vectorised, JAX‑backed vertex enumeration with tolerance deduplication."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    num_facets, dimension = matrix.shape

    if dimension == 0:
        raise ValueError("Polytope dimension must be positive.")

    idx = _all_index_combinations(num_facets, dimension)
    xs, mask = _feasible_solutions(jnp.asarray(matrix), jnp.asarray(offsets), idx, atol)
    if not bool(jnp.any(mask).item()):
        msg = "No vertices found; polytope may be empty or unbounded."
        raise ValueError(msg)

    feasible = jnp.where(mask[:, None], xs, jnp.inf)
    unique = _unique_rows_jax(feasible, atol=atol)
    # Filter out sentinel rows (where any entry is inf).
    unique = unique[jnp.all(jnp.isfinite(unique), axis=1)]
    return unique.astype(jnp.float64)


def remove_redundant_facets(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Vectorised redundancy pruning using matrix operations."""
    matrix, offsets = _shared.validate_halfspace_data(B, c)
    matrix, offsets = _shared.deduplicate_facets(matrix, offsets, atol=atol)
    vertices = enumerate_vertices(matrix, offsets, atol=atol)

    m_j = jnp.asarray(matrix)
    v_j = jnp.asarray(vertices)
    c_j = jnp.asarray(offsets)
    distances = jnp.abs(m_j @ v_j.T - c_j[:, None])
    atol_j = jnp.asarray(atol, dtype=distances.dtype)
    keep = jnp.asarray(jnp.any(distances <= atol_j, axis=1), dtype=bool)

    if not bool(jnp.any(keep).item()):
        msg = "All facets were marked redundant; check the input polytope."
        raise ValueError(msg)

    reduced_B = matrix[keep, :]
    reduced_c = offsets[keep]
    return reduced_B, reduced_c
