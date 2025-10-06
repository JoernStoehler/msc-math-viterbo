"""Shared polytope helpers (JAX-first) reused across implementation variants."""

from __future__ import annotations

import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped.numpy_bytes import fingerprint_halfspace

_POLYTOPE_CACHE_MAX_SIZE: Final[int] = 128
_POLYTOPE_CACHE: "OrderedDict[tuple[str, str], PolytopeCombinatorics]" = OrderedDict()
_POLYTOPE_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class NormalCone:
    """Normal cone data attached to a polytope vertex."""

    vertex: Float[Array, " dimension"]
    active_facets: tuple[int, ...]
    normals: Float[Array, " num_active dimension"]

    def __post_init__(self) -> None:
        """Normalise arrays describing the normal cone."""
        vertex = jnp.asarray(self.vertex, dtype=jnp.float64)
        normals = jnp.asarray(self.normals, dtype=jnp.float64)
        object.__setattr__(self, "vertex", vertex)
        object.__setattr__(self, "normals", normals)


@dataclass(frozen=True)
class PolytopeCombinatorics:
    """Cached combinatorial structure derived from a ``Polytope``."""

    vertices: Float[Array, " num_vertices dimension"]
    facet_adjacency: Array
    normal_cones: tuple[NormalCone, ...]

    def __post_init__(self) -> None:
        """Validate cached arrays for combinatorial reuse."""
        vertices = jnp.asarray(self.vertices, dtype=jnp.float64)
        adjacency = jnp.asarray(self.facet_adjacency, dtype=bool)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "facet_adjacency", adjacency)


@dataclass(frozen=True)
class Polytope:
    """Immutable container describing a convex polytope via half-space data."""

    name: str
    B: Float[Array, " num_facets dimension"]
    c: Float[Array, " num_facets"]
    description: str = ""
    reference_capacity: float | None = None

    def __post_init__(self) -> None:
        """Normalize inputs and freeze the underlying arrays."""
        matrix = jnp.asarray(self.B, dtype=jnp.float64)
        offsets = jnp.asarray(self.c, dtype=jnp.float64)

        if matrix.ndim != 2:
            msg = "Facet matrix B must be two-dimensional."
            raise ValueError(msg)

        if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
            msg = "Offsets vector c must match the number of facets."
            raise ValueError(msg)

        object.__setattr__(self, "B", matrix)
        object.__setattr__(self, "c", offsets)

    @property
    def dimension(self) -> int:
        """Ambient dimension of the polytope."""
        return int(self.B.shape[1])

    @property
    def facets(self) -> int:
        """Number of facet-defining inequalities."""
        return int(self.B.shape[0])

    def halfspace_data(
        self,
    ) -> tuple[
        Float[Array, " num_facets dimension"],
        Float[Array, " num_facets"],
    ]:
        """Return copies of ``(B, c)`` suitable for downstream mutation."""
        return jnp.array(self.B, copy=True), jnp.array(self.c, copy=True)

    def with_metadata(
        self, *, name: str | None = None, description: str | None = None
    ) -> "Polytope":
        """Return a shallow copy with updated metadata while sharing arrays."""
        return Polytope(
            name=name or self.name,
            B=self.B,
            c=self.c,
            description=description or self.description,
            reference_capacity=self.reference_capacity,
        )


def clear_polytope_cache() -> None:
    """Clear all cached polytope combinatorics entries."""
    with _POLYTOPE_CACHE_LOCK:
        _POLYTOPE_CACHE.clear()


def _halfspace_fingerprint(
    matrix: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    decimals: int = 12,
) -> str:
    """Return a deterministic hash for a half-space description."""
    return fingerprint_halfspace(matrix, offsets, decimals=decimals)


def polytope_fingerprint(polytope: Polytope, *, decimals: int = 12) -> str:
    """Return a hash that uniquely identifies ``polytope`` up to rounding."""
    return _halfspace_fingerprint(polytope.B, polytope.c, decimals=decimals)


def _tolerance_fingerprint(atol: float) -> str:
    """Return a deterministic fingerprint for ``atol``."""
    return struct.pack("!d", float(atol)).hex()


def polytope_cache_key(polytope: Polytope, atol: float) -> tuple[str, str]:
    """Return the cache key used for combinatorics lookups."""
    return polytope_fingerprint(polytope), _tolerance_fingerprint(atol)


def cache_enabled(use_cache: bool) -> bool:
    """Return ``True`` if caching should be attempted."""
    disabled = os.environ.get("VITERBO_DISABLE_CACHE", "0") == "1"
    return use_cache and not disabled


def cache_lookup(key: tuple[str, str]) -> PolytopeCombinatorics | None:
    """Return the cached combinatorics value, updating LRU order."""
    with _POLYTOPE_CACHE_LOCK:
        if key not in _POLYTOPE_CACHE:
            return None
        value = _POLYTOPE_CACHE.pop(key)
        _POLYTOPE_CACHE[key] = value
        return value


def cache_store(key: tuple[str, str], value: PolytopeCombinatorics) -> None:
    """Insert ``value`` into the cache while enforcing the size bound."""
    with _POLYTOPE_CACHE_LOCK:
        if key in _POLYTOPE_CACHE:
            _POLYTOPE_CACHE.pop(key)
        _POLYTOPE_CACHE[key] = value
        while len(_POLYTOPE_CACHE) > _POLYTOPE_CACHE_MAX_SIZE:
            _POLYTOPE_CACHE.popitem(last=False)


def build_combinatorics(
    matrix: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    vertices: Float[Array, " num_vertices dimension"],
    *,
    atol: float,
) -> PolytopeCombinatorics:
    """Construct combinatorial data given facets and enumerated vertices."""
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

        # Update adjacency (Python loop with JAX scatter updates for readability).
        active_list = [int(x) for x in active.tolist()]
        for first_index, second_index in combinations(active_list, 2):
            adjacency = adjacency.at[first_index, second_index].set(True)
            adjacency = adjacency.at[second_index, first_index].set(True)

        normals = m[active, :]
        normal_cones.append(
            NormalCone(
                vertex=vertex,
                active_facets=tuple(active_list),
                normals=normals,
            )
        )

    adjacency = adjacency.at[jnp.diag_indices(facet_count)].set(False)
    return PolytopeCombinatorics(
        vertices=vtx,
        facet_adjacency=adjacency,
        normal_cones=tuple(normal_cones),
    )
