"""Combinatorics with caching (datasets layer).

Builds NormalCone and PolytopeCombinatorics for a Polytope, with LRU cache.
"""

from __future__ import annotations

import os
import struct
import threading
from itertools import combinations
from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.datasets.types import NormalCone, Polytope, PolytopeCombinatorics
from viterbo.math.geometry import enumerate_vertices
from viterbo._wrapped.numpy_bytes import fingerprint_halfspace


_POLYTOPE_CACHE_MAX_SIZE: Final[int] = 128
_POLYTOPE_CACHE: "dict[tuple[str, str], PolytopeCombinatorics]" = {}
_POLYTOPE_CACHE_ORDER: "list[tuple[str, str]]" = []
_POLYTOPE_CACHE_LOCK = threading.RLock()


def _halfspace_fingerprint(
    matrix: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    decimals: int = 12,
) -> str:
    return fingerprint_halfspace(matrix, offsets, decimals=decimals)


def polytope_fingerprint(polytope: Polytope, *, decimals: int = 12) -> str:
    """Return a stable hex digest for the polytope half-space representation."""
    normals, offsets = polytope.halfspace_data()
    return _halfspace_fingerprint(normals, offsets, decimals=decimals)


def _tolerance_fingerprint(atol: float) -> str:
    return struct.pack("!d", float(atol)).hex()


def polytope_cache_key(polytope: Polytope, atol: float) -> tuple[str, str]:
    return polytope_fingerprint(polytope), _tolerance_fingerprint(atol)


def clear_polytope_cache() -> None:
    """Clear the in-process combinatorics cache (LRU)."""
    with _POLYTOPE_CACHE_LOCK:
        _POLYTOPE_CACHE.clear()
        _POLYTOPE_CACHE_ORDER.clear()


def _cache_enabled(use_cache: bool) -> bool:
    disabled = os.environ.get("VITERBO_DISABLE_CACHE", "0") == "1"
    return use_cache and not disabled


def _cache_lookup(key: tuple[str, str]) -> PolytopeCombinatorics | None:
    with _POLYTOPE_CACHE_LOCK:
        if key not in _POLYTOPE_CACHE:
            return None
        _POLYTOPE_CACHE_ORDER.remove(key)
        _POLYTOPE_CACHE_ORDER.append(key)
        return _POLYTOPE_CACHE[key]


def _cache_store(key: tuple[str, str], value: PolytopeCombinatorics) -> None:
    with _POLYTOPE_CACHE_LOCK:
        if key in _POLYTOPE_CACHE:
            _POLYTOPE_CACHE[key] = value
            try:
                _POLYTOPE_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            _POLYTOPE_CACHE_ORDER.append(key)
        else:
            _POLYTOPE_CACHE[key] = value
            _POLYTOPE_CACHE_ORDER.append(key)
        while len(_POLYTOPE_CACHE_ORDER) > _POLYTOPE_CACHE_MAX_SIZE:
            oldest = _POLYTOPE_CACHE_ORDER.pop(0)
            _POLYTOPE_CACHE.pop(oldest, None)


def _build_combinatorics(
    matrix: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    vertices: Float[Array, " num_vertices dimension"],
    *,
    atol: float,
) -> PolytopeCombinatorics:
    m = jnp.asarray(matrix)
    c = jnp.asarray(offsets)
    vtx = jnp.asarray(vertices)
    facet_count = int(m.shape[0])
    adjacency = jnp.zeros((facet_count, facet_count), dtype=bool)
    normal_cones: list[NormalCone] = []

    for k in range(int(vtx.shape[0])):
        vertex = vtx[k]
        residuals = m @ vertex - c
        active = jnp.where(jnp.abs(residuals) <= float(atol))[0]
        if int(active.size) == 0:
            continue
        active_list = [int(x) for x in active.tolist()]
        for first_index, second_index in combinations(active_list, 2):
            adjacency = adjacency.at[first_index, second_index].set(True)
            adjacency = adjacency.at[second_index, first_index].set(True)
        normals = m[active, :]
        normal_cones.append(
            NormalCone(vertex=vertex, active_facets=tuple(active_list), normals=normals)
        )

    adjacency = adjacency.at[jnp.diag_indices(facet_count)].set(False)
    return PolytopeCombinatorics(
        vertices=vtx, facet_adjacency=adjacency, normal_cones=tuple(normal_cones)
    )


def polytope_combinatorics(
    polytope: Polytope,
    *,
    atol: float = 1e-9,
    use_cache: bool = True,
) -> PolytopeCombinatorics:
    """Return vertices, facet adjacency, and normal cones for `polytope`."""
    key = polytope_cache_key(polytope, atol)
    if _cache_enabled(use_cache):
        cached = _cache_lookup(key)
        if cached is not None:
            return cached
    B, c = polytope.halfspace_data()
    verts = enumerate_vertices(B, c, atol=atol)
    combinatorics = _build_combinatorics(B, c, verts, atol=atol)
    if _cache_enabled(use_cache):
        _cache_store(key, combinatorics)
    return combinatorics


__all__ = [
    "polytope_combinatorics",
    "polytope_fingerprint",
    "clear_polytope_cache",
]
