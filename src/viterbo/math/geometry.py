"""Half-space and vertex utilities (pure math layer).

Functions operate on JAX arrays and return arrays or tuples of arrays.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped.spatial import convex_hull_equations


def _validate_halfspace_data(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    matrix = jnp.asarray(B, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)
    if matrix.ndim != 2:
        raise ValueError("Facet matrix B must be two-dimensional.")
    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        raise ValueError("Offset vector c must match the number of facets.")
    return matrix, offsets


def _unique_rows(
    points: Float[Array, " num_points dimension"],
    *,
    atol: float,
) -> Float[Array, " num_unique dimension"]:
    v = jnp.asarray(points)
    if v.size == 0:
        return v
    order = jnp.lexsort(v.T)
    sorted_pts = v[order]

    def step(carry: jnp.ndarray, current: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        last = carry
        keep = jnp.any(jnp.abs(current - last) > float(atol))
        new_last = jax.lax.select(keep, current, last)
        return new_last, keep

    init = sorted_pts[0]
    _, keep_rest = jax.lax.scan(lambda carr, cur: step(carr, cur), init, sorted_pts[1:])
    keep_mask = jnp.concatenate((jnp.array([True]), keep_rest))
    return jnp.compress(keep_mask, sorted_pts, axis=0)


def _deduplicate_facets(
    matrix: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float,
) -> tuple[Float[Array, " num_unique_facets dimension"], Float[Array, " num_unique_facets"]]:
    m = jnp.asarray(matrix)
    c = jnp.asarray(offsets)
    eq_rows = jnp.all(jnp.abs(m[:, None, :] - m[None, :, :]) <= float(atol), axis=2)
    eq_offsets = jnp.abs(c[:, None] - c[None, :]) <= float(atol)
    eq = jnp.logical_and(eq_rows, eq_offsets)
    earlier = jnp.tril(eq, k=-1)
    keep = jnp.logical_not(jnp.any(earlier, axis=1))
    return m[keep, :], c[keep]


def enumerate_vertices(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[Array, " num_vertices dimension"]:
    """Enumerate vertices of a bounded polytope ``{x | Bx ≤ c}``."""
    matrix_np, offsets_np = _validate_halfspace_data(B, c)
    num_facets, dimension = matrix_np.shape
    if dimension == 0:
        raise ValueError("Polytope dimension must be positive.")
    matrix = jnp.asarray(matrix_np)
    offsets = jnp.asarray(offsets_np)

    vertices: list[jnp.ndarray] = []
    combinations_iter = cast(Iterable[tuple[int, ...]], combinations(range(num_facets), dimension))
    for combo in combinations_iter:
        indices = tuple(combo)
        subset = matrix[jnp.array(indices), :]
        s = jnp.linalg.svd(subset, compute_uv=False)
        rank = int((s > (jnp.max(s) * 1e-12)).sum())
        if rank < dimension:
            continue
        subset_offsets = offsets[jnp.array(indices)]
        solution = jnp.linalg.solve(subset, subset_offsets)
        feasible = bool(jnp.all(matrix @ solution <= offsets + float(atol)).item())
        if feasible:
            vertices.append(solution)
    if not vertices:
        raise ValueError("No vertices found; polytope may be empty or unbounded.")
    stacked = jnp.stack(vertices, axis=0)
    return _unique_rows(stacked, atol=atol)


def remove_redundant_facets(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Prune redundant inequalities from a half-space description."""
    m, cvec = _validate_halfspace_data(B, c)
    m, cvec = _deduplicate_facets(m, cvec, atol=atol)
    v = enumerate_vertices(m, cvec, atol=atol)
    keep_mask: list[bool] = []
    for i in range(int(m.shape[0])):
        row = m[i]
        offset = cvec[i]
        distances = jnp.abs(v @ row - offset)
        keep_mask.append(bool(jnp.any(distances <= float(atol)).item()))
    if not any(keep_mask):
        raise ValueError("All facets were marked redundant; check the input polytope.")
    keep = jnp.asarray(keep_mask, dtype=bool)
    reduced_B = m[keep, :]
    reduced_c = cvec[keep]
    return reduced_B, reduced_c


def vertices_from_halfspaces(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[Array, " num_vertices dimension"]:
    """Return vertices of a bounded polytope defined by half-spaces Bx ≤ c."""
    return enumerate_vertices(B, c, atol=atol)


def halfspaces_from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
    *,
    qhull_options: str | None = None,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Return a reduced half-space description (B, c) of the convex hull."""
    equations = convex_hull_equations(vertices, qhull_options=qhull_options)
    normals = equations[:, :-1]
    offsets = equations[:, -1]
    B_j = jnp.asarray(normals, dtype=jnp.float64)
    c_j = jnp.asarray(-offsets, dtype=jnp.float64)
    return remove_redundant_facets(B_j, c_j)
