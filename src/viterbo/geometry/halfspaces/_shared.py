"""Shared utilities for half-space representations (JAX-first)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def validate_halfspace_data(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Validate and normalize half-space inputs (returns JAX arrays)."""
    matrix = jnp.asarray(B, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)

    if matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        msg = "Offset vector c must match the number of facets."
        raise ValueError(msg)

    return matrix, offsets


def unique_rows(
    points: Float[Array, " num_points dimension"],
    *,
    atol: float,
) -> Float[Array, " num_unique dimension"]:
    """Deduplicate stacked vectors using an infinity-norm tolerance (JAX)."""
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


def deduplicate_facets(
    matrix: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float,
) -> tuple[Float[Array, " num_unique_facets dimension"], Float[Array, " num_unique_facets"]]:
    """Remove near-duplicate facet rows with shared offsets (JAX)."""
    m = jnp.asarray(matrix)
    c = jnp.asarray(offsets)
    eq_rows = jnp.all(jnp.abs(m[:, None, :] - m[None, :, :]) <= float(atol), axis=2)
    eq_offsets = jnp.abs(c[:, None] - c[None, :]) <= float(atol)
    eq = jnp.logical_and(eq_rows, eq_offsets)
    earlier = jnp.tril(eq, k=-1)
    keep = jnp.logical_not(jnp.any(earlier, axis=1))
    return m[keep, :], c[keep]
