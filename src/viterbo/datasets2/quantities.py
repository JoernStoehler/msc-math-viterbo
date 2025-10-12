"""Unified wrappers around :mod:`viterbo.math` quantities."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.math import geometry, volume


def vertices(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    method: Literal["enumerate"] = "enumerate",
    atol: float = 1e-9,
) -> Float[Array, " num_vertices dimension"]:
    """Return vertices of ``{x | normals @ x ≤ offsets}`` using the requested method."""

    normals64 = jnp.asarray(normals, dtype=jnp.float64)
    offsets64 = jnp.asarray(offsets, dtype=jnp.float64)
    if method == "enumerate":
        return geometry.enumerate_vertices(normals64, offsets64, atol=atol)
    raise ValueError(f"Unknown vertex enumeration method: {method}")


def halfspaces(
    vertices_data: Float[Array, " num_vertices dimension"],
    *,
    method: Literal["qhull"] = "qhull",
    qhull_options: str | None = None,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    """Return a half-space representation of the convex hull of ``vertices``."""

    verts64 = jnp.asarray(vertices_data, dtype=jnp.float64)
    if method == "qhull":
        return geometry.halfspaces_from_vertices(verts64, qhull_options=qhull_options)
    raise ValueError(f"Unknown half-space recovery method: {method}")


def volume_from_vertices(
    vertices_data: Float[Array, " num_vertices dimension"],
    *,
    method: Literal["reference"] = "reference",
) -> float:
    """Return a volume estimate from vertex data."""

    verts64 = jnp.asarray(vertices_data, dtype=jnp.float64)
    if method == "reference":
        return float(volume.volume_reference(verts64))
    raise ValueError(f"Unknown vertex volume method: {method}")


def volume_from_halfspaces(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    method: Literal["reference", "fast"] = "reference",
    atol: float = 1e-9,
) -> float:
    """Return the convex volume according to ``method``."""

    normals64 = jnp.asarray(normals, dtype=jnp.float64)
    offsets64 = jnp.asarray(offsets, dtype=jnp.float64)
    if method == "reference":
        return float(volume.polytope_volume_reference(normals64, offsets64, atol=atol))
    if method == "fast":
        return float(volume.polytope_volume_fast(normals64, offsets64, atol=atol))
    raise ValueError(f"Unknown volume method: {method}")


__all__ = ["vertices", "halfspaces", "volume_from_vertices", "volume_from_halfspaces"]
