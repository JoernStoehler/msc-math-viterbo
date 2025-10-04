"""Shared polytope helpers reused across implementation variants."""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from typing import Final

import numpy as np
from jaxtyping import Float

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_FACET_MATRIX_AXES: Final[str] = f"{_FACET_AXIS} {_DIMENSION_AXIS}"
_SQUARE_MATRIX_AXES: Final[str] = f"{_DIMENSION_AXIS} {_DIMENSION_AXIS}"
_VERTEX_MATRIX_AXES: Final[str] = "num_vertices dimension"

_POLYTOPE_CACHE_MAX_SIZE: Final[int] = 128
_POLYTOPE_CACHE: "OrderedDict[tuple[str, str], PolytopeCombinatorics]" = OrderedDict()
_POLYTOPE_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class NormalCone:
    """Normal cone data attached to a polytope vertex."""

    vertex: Float[np.ndarray, _DIMENSION_AXIS]
    active_facets: tuple[int, ...]
    normals: Float[np.ndarray, " num_active dimension"]

    def __post_init__(self) -> None:
        """Normalise arrays describing the normal cone."""
        vertex = np.asarray(self.vertex, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        object.__setattr__(self, "vertex", vertex)
        object.__setattr__(self, "normals", normals)
        vertex.setflags(write=False)
        normals.setflags(write=False)


@dataclass(frozen=True)
class PolytopeCombinatorics:
    """Cached combinatorial structure derived from a ``Polytope``."""

    vertices: Float[np.ndarray, _VERTEX_MATRIX_AXES]
    facet_adjacency: np.ndarray
    normal_cones: tuple[NormalCone, ...]

    def __post_init__(self) -> None:
        """Validate cached arrays for combinatorial reuse."""
        vertices = np.asarray(self.vertices, dtype=float)
        adjacency = np.asarray(self.facet_adjacency, dtype=bool)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "facet_adjacency", adjacency)
        vertices.setflags(write=False)
        adjacency.setflags(write=False)


@dataclass(frozen=True)
class Polytope:
    """Immutable container describing a convex polytope via half-space data."""

    name: str
    B: Float[np.ndarray, _FACET_MATRIX_AXES]
    c: Float[np.ndarray, _FACET_AXIS]
    description: str = ""
    reference_capacity: float | None = None

    def __post_init__(self) -> None:
        """Normalize inputs and freeze the underlying arrays."""
        matrix = np.asarray(self.B, dtype=float)
        offsets = np.asarray(self.c, dtype=float)

        if matrix.ndim != 2:
            msg = "Facet matrix B must be two-dimensional."
            raise ValueError(msg)

        if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
            msg = "Offsets vector c must match the number of facets."
            raise ValueError(msg)

        object.__setattr__(self, "B", matrix)
        object.__setattr__(self, "c", offsets)

        matrix.setflags(write=False)
        offsets.setflags(write=False)

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
        Float[np.ndarray, _FACET_MATRIX_AXES],
        Float[np.ndarray, _FACET_AXIS],
    ]:
        """Return copies of ``(B, c)`` suitable for downstream mutation."""
        return np.array(self.B, copy=True), np.array(self.c, copy=True)

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
    matrix: Float[np.ndarray, _FACET_MATRIX_AXES],
    offsets: Float[np.ndarray, _FACET_AXIS],
    *,
    decimals: int = 12,
) -> str:
    """Return a deterministic hash for a half-space description."""
    rounded_matrix = np.round(np.asarray(matrix, dtype=float), decimals=decimals)
    rounded_offsets = np.round(np.asarray(offsets, dtype=float), decimals=decimals)

    contiguous_matrix = np.ascontiguousarray(rounded_matrix)
    contiguous_offsets = np.ascontiguousarray(rounded_offsets)

    hasher = hashlib.sha256()
    hasher.update(np.array(contiguous_matrix.shape, dtype=np.int64).tobytes())
    hasher.update(np.array(contiguous_offsets.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous_matrix.tobytes())
    hasher.update(contiguous_offsets.tobytes())
    return hasher.hexdigest()


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
    matrix: Float[np.ndarray, _FACET_MATRIX_AXES],
    offsets: Float[np.ndarray, _FACET_AXIS],
    vertices: Float[np.ndarray, _VERTEX_MATRIX_AXES],
    *,
    atol: float,
) -> PolytopeCombinatorics:
    """Construct combinatorial data given facets and enumerated vertices."""
    facet_count = matrix.shape[0]
    adjacency = np.zeros((facet_count, facet_count), dtype=bool)
    normal_cones: list[NormalCone] = []

    for vertex in vertices:
        residuals = matrix @ vertex - offsets
        active = np.where(np.abs(residuals) <= atol)[0]
        if active.size == 0:
            continue

        for first_index, second_index in combinations(active, 2):
            adjacency[first_index, second_index] = True
            adjacency[second_index, first_index] = True

        normals = matrix[active, :]
        normal_cones.append(
            NormalCone(
                vertex=vertex,
                active_facets=tuple(int(index) for index in active),
                normals=normals,
            )
        )

    np.fill_diagonal(adjacency, False)
    return PolytopeCombinatorics(
        vertices=np.asarray(vertices, dtype=float),
        facet_adjacency=adjacency,
        normal_cones=tuple(normal_cones),
    )
