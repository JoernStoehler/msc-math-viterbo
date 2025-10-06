"""Reference implementations for half-space utilities (readable, trusted, JAX-first)."""

from __future__ import annotations

import itertools
from typing import Iterable, cast

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.geometry.halfspaces import _shared


def enumerate_vertices(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[Array, " num_vertices dimension"]:
    """Enumerate vertices of a bounded polytope ``{x | Bx ≤ c}``.

    Uses ``jax.numpy`` for linear algebra while keeping Python control flow for
    readability. Falls back to NumPy arrays only at return time.
    """
    matrix_np, offsets_np = _shared.validate_halfspace_data(B, c)
    num_facets, dimension = matrix_np.shape

    if dimension == 0:
        raise ValueError("Polytope dimension must be positive.")

    matrix = jnp.asarray(matrix_np)
    offsets = jnp.asarray(offsets_np)

    vertices: list[jnp.ndarray] = []
    combinations_iter = cast(
        Iterable[tuple[int, ...]], itertools.combinations(range(num_facets), dimension)
    )
    for combination in combinations_iter:
        indices = tuple(combination)
        subset = matrix[jnp.array(indices), :]
        # JAX-first rank check via SVD (readable and robust).
        s = jnp.linalg.svd(subset, compute_uv=False)
        rank = int((s > (jnp.max(s) * 1e-12)).sum())
        if rank < dimension:
            continue

        subset_offsets = offsets[jnp.array(indices)]
        # Solve (subset assumed full rank by the SVD rank test)
        solution = jnp.linalg.solve(subset, subset_offsets)

        feasible = bool(jnp.all(matrix @ solution <= offsets + float(atol)).item())
        if feasible:
            vertices.append(solution)

    if not vertices:
        msg = "No vertices found; polytope may be empty or unbounded."
        raise ValueError(msg)

    stacked = jnp.stack(vertices, axis=0)
    return _shared.unique_rows(stacked, atol=atol)


def remove_redundant_facets(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Prune redundant inequalities from a half-space description.

    Computes distances with ``jax.numpy`` and uses a Python loop for clarity.
    """
    m, cvec = _shared.validate_halfspace_data(B, c)
    m, cvec = _shared.deduplicate_facets(m, cvec, atol=atol)
    v = enumerate_vertices(m, cvec, atol=atol)

    keep_mask: list[bool] = []
    for i in range(int(m.shape[0])):
        row = m[i]
        offset = cvec[i]
        distances = jnp.abs(v @ row - offset)
        keep_mask.append(bool(jnp.any(distances <= float(atol)).item()))

    if not any(keep_mask):
        msg = "All facets were marked redundant; check the input polytope."
        raise ValueError(msg)

    keep = jnp.asarray(keep_mask, dtype=bool)
    reduced_B = m[keep, :]
    reduced_c = cvec[keep]
    return reduced_B, reduced_c
