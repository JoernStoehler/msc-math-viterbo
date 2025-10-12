"""Flat geometry compatibility helpers for the modern viterbo API.

This module consolidates the minimal functionality needed from the legacy
`viterbo.geometry` package so that modern capacity/spectrum code can depend on
flat `viterbo.*` imports. It provides:

- A lightweight `Polytope` dataclass for half-space data and metadata.
- Combinatorics cache and builders: `NormalCone`, `PolytopeCombinatorics`,
  and `polytope_combinatorics`.
- Half-space utilities: `enumerate_vertices`, `remove_redundant_facets`,
  `vertices_from_halfspaces`, `halfspaces_from_vertices`.
- Canonical polytope families and transforms used by tests and benchmarks:
  `hypercube`, `cross_polytope`, `regular_polygon_product`, `cartesian_product`,
  `simplex_with_uniform_weights`, `truncated_simplex_four_dim`,
  `viterbo_counterexample`, `catalog`, and random transforms.
- Volume helpers matching prior call sites: `polytope_volume_reference`,
  `polytope_volume_fast`.

The implementations are JAX-first and reuse SciPy interop via
`viterbo._wrapped.spatial` where necessary.
"""

from __future__ import annotations

import math
import os
import struct
import threading
from dataclasses import dataclass
from itertools import combinations, product
from typing import Final, Iterable, Sequence, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped.spatial import (
    QhullError,
    convex_hull_equations,
    convex_hull_volume,
    delaunay_simplices,
)


# -----------------------------------------------------------------------------
# Dataclasses and combinatorics cache (ported from legacy geometry._shared)


@dataclass(frozen=True)
class NormalCone:
    """Normal cone data attached to a polytope vertex."""

    vertex: Float[Array, " dimension"]
    active_facets: tuple[int, ...]
    normals: Float[Array, " num_active dimension"]

    def __post_init__(self) -> None:
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
        matrix = jnp.asarray(self.B, dtype=jnp.float64)
        offsets = jnp.asarray(self.c, dtype=jnp.float64)
        if matrix.ndim != 2:
            raise ValueError("Facet matrix B must be two-dimensional.")
        if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
            raise ValueError("Offsets vector c must match the number of facets.")
        object.__setattr__(self, "B", matrix)
        object.__setattr__(self, "c", offsets)

    @property
    def dimension(self) -> int:  # noqa: D401 (concise)
        return int(self.B.shape[1])

    @property
    def facets(self) -> int:  # noqa: D401 (concise)
        return int(self.B.shape[0])

    def halfspace_data(
        self,
    ) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
        return jnp.array(self.B, copy=True), jnp.array(self.c, copy=True)

    def with_metadata(self, *, name: str | None = None, description: str | None = None) -> "Polytope":
        return Polytope(
            name=name or self.name,
            B=self.B,
            c=self.c,
            description=description or self.description,
            reference_capacity=self.reference_capacity,
        )


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
    from viterbo._wrapped.numpy_bytes import fingerprint_halfspace

    return fingerprint_halfspace(matrix, offsets, decimals=decimals)


def polytope_fingerprint(polytope: Polytope, *, decimals: int = 12) -> str:
    return _halfspace_fingerprint(polytope.B, polytope.c, decimals=decimals)


def _tolerance_fingerprint(atol: float) -> str:
    return struct.pack("!d", float(atol)).hex()


def polytope_cache_key(polytope: Polytope, atol: float) -> tuple[str, str]:
    return polytope_fingerprint(polytope), _tolerance_fingerprint(atol)


def clear_polytope_cache() -> None:
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
        # bump LRU order
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


# -----------------------------------------------------------------------------
# Half-space utilities (ported from legacy geometry.halfspaces)


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
    combinations_iter = cast(
        Iterable[tuple[int, ...]], combinations(range(num_facets), dimension)
    )
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
    return enumerate_vertices(B, c, atol=atol)


def halfspaces_from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
    *,
    qhull_options: str | None = None,
) -> tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]]:
    equations = convex_hull_equations(vertices, qhull_options=qhull_options)
    normals = equations[:, :-1]
    offsets = equations[:, -1]
    B_j = jnp.asarray(normals, dtype=jnp.float64)
    c_j = jnp.asarray(-offsets, dtype=jnp.float64)
    return remove_redundant_facets(B_j, c_j)


def polytope_combinatorics(
    polytope: Polytope,
    *,
    atol: float = 1e-9,
    use_cache: bool = True,
) -> PolytopeCombinatorics:
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


# -----------------------------------------------------------------------------
# Volume helpers (reference + fast)


def _volume_of_simplices(simplex_vertices: Float[Array, " num_simplices vertices dimension"]) -> float:
    v = jnp.asarray(simplex_vertices)
    base = v[:, 0, :]
    edges = v[:, 1:, :] - base[:, None, :]
    determinants = jnp.linalg.det(edges)
    dimension = v.shape[2]
    total = jnp.sum(jnp.abs(determinants)) / math.factorial(int(dimension))
    return float(total)


def polytope_volume_reference(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    vertices = enumerate_vertices(B, c, atol=atol)
    return convex_hull_volume(vertices, qhull_options="QJ")


def polytope_volume_fast(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> float:
    vertices = enumerate_vertices(B, c, atol=atol)
    try:
        simplices = delaunay_simplices(vertices, qhull_options="QJ")
    except QhullError:
        return convex_hull_volume(vertices, qhull_options="QJ")
    return _volume_of_simplices(vertices[simplices])


# -----------------------------------------------------------------------------
# Canonical polytope families and transformations (subset used in tests)


def cartesian_product(
    first: Polytope,
    second: Polytope,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    B1, c1 = first.halfspace_data()
    B2, c2 = second.halfspace_data()
    upper = jnp.hstack((B1, jnp.zeros((B1.shape[0], B2.shape[1]))))
    lower = jnp.hstack((jnp.zeros((B2.shape[0], B1.shape[1])), B2))
    B = jnp.vstack((upper, lower))
    c = jnp.concatenate((c1, c2))
    product_name = name or f"{first.name}x{second.name}"
    product_description = description or (
        "Cartesian product constructed from "
        f"{first.name} (dim {first.dimension}) and {second.name} (dim {second.dimension})."
    )
    return Polytope(name=product_name, B=B, c=c, description=product_description)


def affine_transform(
    polytope: Polytope,
    matrix: Float[Array, " dimension dimension"],
    *,
    translation: Float[Array, " dimension"] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    if matrix.shape != (polytope.dimension, polytope.dimension):
        raise ValueError("Linear transform must match the ambient dimension.")
    det = jnp.linalg.det(matrix)
    if jnp.isclose(det, 0.0):
        raise ValueError("Affine transform requires an invertible matrix.")
    matrix_inv = jnp.linalg.inv(matrix)
    translation_vec = (
        jnp.zeros(polytope.dimension)
        if translation is None
        else jnp.asarray(translation, dtype=jnp.float64)
    )
    if translation_vec.shape != (polytope.dimension,):
        raise ValueError("Translation vector must match the ambient dimension.")
    B_transformed = polytope.B @ matrix_inv
    c_transformed = polytope.c + B_transformed @ translation_vec
    return Polytope(
        name=name or f"{polytope.name}-affine",
        B=B_transformed,
        c=c_transformed,
        description=description
        or (
            f"Affine image of {polytope.name} via matrix with det {float(det):.3f}."
        ),
        reference_capacity=polytope.reference_capacity,
    )


def translate_polytope(
    polytope: Polytope,
    translation: Float[Array, " dimension"],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    return affine_transform(
        polytope,
        jnp.eye(polytope.dimension),
        translation=translation,
        name=name or f"{polytope.name}-translated",
        description=description or f"Translation of {polytope.name}.",
    )


def mirror_polytope(
    polytope: Polytope,
    axes: Sequence[bool],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    if len(tuple(axes)) != polytope.dimension:
        raise ValueError("Axis mask must match the polytope dimension.")
    signs = jnp.where(jnp.asarray(tuple(axes), dtype=bool), -1.0, 1.0)
    matrix = jnp.diag(signs)
    return affine_transform(
        polytope,
        matrix,
        translation=jnp.zeros(polytope.dimension),
        name=name or f"{polytope.name}-mirrored",
        description=description or "Coordinate reflection of the base polytope.",
    )


def rotate_polytope(
    polytope: Polytope,
    *,
    plane: tuple[int, int],
    angle: float,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    i, j = plane
    dimension = polytope.dimension
    if not (0 <= i < dimension and 0 <= j < dimension) or i == j:
        raise ValueError("Rotation plane indices must be distinct and within range.")
    rotation = jnp.eye(dimension)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = rotation.at[i, i].set(cosine)
    rotation = rotation.at[j, j].set(cosine)
    rotation = rotation.at[i, j].set(-sine)
    rotation = rotation.at[j, i].set(sine)
    return affine_transform(
        polytope,
        rotation,
        translation=jnp.zeros(dimension),
        name=name or f"{polytope.name}-rot",
        description=description or f"Rotation of {polytope.name} in plane {(i, j)} by {angle} rad.",
    )


def _regular_polygon_normals(sides: int) -> jnp.ndarray:
    if sides < 3:
        raise ValueError("A polygon requires at least three sides.")
    angles = 2 * jnp.pi * (jnp.arange(sides) / sides)
    normals = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
    return normals


def _rotation_matrix(angle: float) -> jnp.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return jnp.array([[cosine, -sine], [sine, cosine]])


def regular_polygon_product(
    sides_first: int,
    sides_second: int,
    *,
    rotation: float = 0.0,
    radius_first: float = 1.0,
    radius_second: float | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    normals_first = _regular_polygon_normals(sides_first)
    normals_second = _regular_polygon_normals(sides_second)
    if radius_second is None:
        radius_second = radius_first
    rotation_matrix = _rotation_matrix(rotation)
    rotated_second = normals_second @ rotation_matrix.T
    zero_block = jnp.zeros((normals_first.shape[0], 2))
    B_upper = jnp.hstack((normals_first, zero_block))
    B_lower = jnp.hstack((jnp.zeros((rotated_second.shape[0], 2)), rotated_second))
    B_np = jnp.vstack((B_upper, B_lower))
    c_np = jnp.concatenate(
        (
            jnp.full(normals_first.shape[0], float(radius_first)),
            jnp.full(rotated_second.shape[0], float(radius_second)),
        )
    )
    default_name = name or (
        f"{sides_first}gonx{sides_second}gon-rot{int(round(math.degrees(rotation)))}"
    )
    default_description = description or (
        "Product of two regular polygons, yielding a 4D polytope with"
        f" {normals_first.shape[0] + rotated_second.shape[0]} facets."
    )
    return Polytope(
        name=default_name,
        B=jnp.asarray(B_np, dtype=jnp.float64),
        c=jnp.asarray(c_np, dtype=jnp.float64),
        description=default_description,
    )


def cross_polytope(
    dimension: int,
    *,
    radius: float = 1.0,
    name: str | None = None,
) -> Polytope:
    if dimension < 2:
        raise ValueError("Dimension must be at least two.")
    normals = jnp.asarray(list(product((-1.0, 1.0), repeat=dimension)), dtype=jnp.float64)
    c = jnp.full(normals.shape[0], float(radius))
    description = "Centrally symmetric cross-polytope with L1 ball geometry."
    return Polytope(
        name=name or f"cross-polytope-{dimension}d",
        B=normals,
        c=c,
        description=description,
    )


def hypercube(
    dimension: int,
    *,
    radius: float = 1.0,
    name: str | None = None,
) -> Polytope:
    if dimension < 2:
        raise ValueError("Dimension must be at least two.")
    identity = jnp.eye(dimension)
    B_matrix = jnp.vstack((identity, -identity))
    c = jnp.full(2 * dimension, float(radius))
    description = "Hypercube aligned with the coordinate axes."
    return Polytope(
        name=name or f"hypercube-{dimension}d",
        B=B_matrix,
        c=c,
        description=description,
    )


def simplex_with_uniform_weights(
    dimension: int,
    *,
    last_offset: float | None = None,
    name: str | None = None,
) -> Polytope:
    if dimension < 2:
        raise ValueError("Dimension must be at least two.")
    B_matrix = jnp.eye(dimension)
    extra = -jnp.ones((1, dimension))
    B_matrix = jnp.vstack((B_matrix, extra))
    offsets = jnp.ones(dimension + 1)
    if last_offset is None:
        last_offset = dimension / 2
    offsets = offsets.at[-1].set(float(last_offset))
    polytope_name = name or f"uniform-simplex-{dimension}d"
    reference_capacity = None
    if dimension == 4 and math.isclose(last_offset, dimension / 2):
        reference_capacity = 9.0
    description = (
        "4D simplex with symmetric coordinates; used as a canonical regression example"
        if dimension == 4
        else "Simplex with uniform coordinate facets and a balancing facet."
    )
    return Polytope(
        name=polytope_name,
        B=B_matrix,
        c=offsets,
        description=description,
        reference_capacity=reference_capacity,
    )


def truncated_simplex_four_dim() -> Polytope:
    B_matrix = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    c = jnp.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.2])
    # Use modern symplectic helper through flat import
    from viterbo.symplectic import standard_symplectic_matrix as _J

    def _haim_kislev_action(B: jnp.ndarray, c_vec: jnp.ndarray) -> float:
        # Minimal embedded action computation for fixed subset/order pair
        order = (2, 0, 4, 3, 1)
        subset = (0, 1, 2, 3, 4)
        m = len(subset)
        B_subset = B[jnp.array(subset), :]
        system = jnp.vstack((
            jnp.concatenate((jnp.ones((1, m)), jnp.zeros((m - 1, m))), axis=0),
            jnp.zeros((m, m)),
        ))[:m, :]
        rhs = jnp.zeros(m)
        rhs = rhs.at[0].set(1.0)
        beta = jnp.linalg.solve(system, rhs)
        J = _J(B.shape[1])
        symp = (B_subset @ jnp.asarray(J)) @ B_subset.T
        total = 0.0
        for i in range(1, m):
            idx_i = order[i]
            wi = beta[idx_i]
            if wi <= 0.0:
                continue
            row = symp[idx_i]
            for j in range(i):
                idx_j = order[j]
                wj = beta[idx_j]
                if wj <= 0.0:
                    continue
                total += wi * wj * row[idx_j]
        if total <= 0.0:
            raise ValueError("Facet ordering yielded a non-positive action.")
        return 0.5 / float(total)

    reference_capacity = _haim_kislev_action(B_matrix, c)
    description = "Simplex-like model with an extra facet; preserves the optimal Reeb action"
    return Polytope(
        name="truncated-simplex-4d",
        B=B_matrix,
        c=c,
        description=description,
        reference_capacity=reference_capacity,
    )


def viterbo_counterexample(radius: float = 1.0) -> Polytope:
    description = (
        "Product of a regular pentagon with its quarter-turned copy,"
        " the Chaidez–Hutchings counterexample to Viterbo's conjecture."
    )
    return regular_polygon_product(
        5,
        5,
        rotation=math.pi / 2,
        radius_first=radius,
        radius_second=radius,
        name="viterbo-counterexample",
        description=description,
    )


def random_affine_map(
    dimension: int,
    *,
    key: Array,
    scale_range: tuple[float, float] = (0.6, 1.4),
    shear_scale: float = 0.25,
    translation_scale: float = 0.3,
) -> tuple[Float[Array, " dimension dimension"], Float[Array, " dimension"]]:
    lower, upper = scale_range
    if lower <= 0 or upper <= 0:
        raise ValueError("Scale factors must be positive.")
    k = key
    for _ in range(32):
        k, k_q, k_scales, k_shear, k_trans = jax.random.split(
            jax.random.PRNGKey(jax.random.randint(k, (), 0, 2**31 - 1, dtype=jnp.uint32).item()), 5
        )
        q_input = jax.random.normal(k_q, (dimension, dimension), dtype=jnp.float64)
        q, _ = jnp.linalg.qr(q_input)
        scales = jax.random.uniform(
            k_scales, (dimension,), minval=lower, maxval=upper, dtype=jnp.float64
        )
        shear_noise = jax.random.normal(k_shear, (dimension, dimension), dtype=jnp.float64)
        shear = jnp.eye(dimension) + shear_noise * shear_scale
        matrix = q @ jnp.diag(scales) @ shear
        if bool(jnp.isclose(jnp.linalg.det(matrix), 0.0).item()):
            continue
        translation = (
            jax.random.normal(k_trans, (dimension,), dtype=jnp.float64) * translation_scale
        )
        return matrix, translation
    raise RuntimeError("Failed to sample an invertible affine map.")


def random_polytope(
    dimension: int,
    *,
    key: Array,
    facets: int | None = None,
    offset_range: tuple[float, float] = (0.5, 1.5),
    translation_scale: float = 0.2,
    name: str | None = None,
    description: str | None = None,
    max_attempts: int = 64,
) -> Polytope:
    if dimension <= 0:
        raise ValueError("Dimension must be positive.")
    low, high = offset_range
    if low <= 0 or high <= 0 or high <= low:
        raise ValueError("Offsets must satisfy 0 < low < high.")
    if facets is None:
        facets = max(dimension + 1, 4 * dimension)
    if facets < dimension + 1:
        raise ValueError("At least dimension + 1 facets are required.")
    identity = jnp.eye(dimension)
    k = key
    for attempt in range(max_attempts):
        k, k_normals, k_offsets, k_trans = jax.random.split(
            jax.random.PRNGKey(jax.random.randint(k, (), 0, 2**31 - 1, dtype=jnp.uint32).item()), 4
        )
        normals = jax.random.normal(k_normals, (facets, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.clip(norms, a_min=1e-12, a_max=None)
        offsets = jax.random.uniform(k_offsets, (facets,), minval=low, maxval=high, dtype=jnp.float64)
        normals = jnp.vstack((normals, identity, -identity))
        offsets = jnp.concatenate((offsets, jnp.full(2 * dimension, float(high))))
        try:
            reduced_B, reduced_c = remove_redundant_facets(normals, offsets, atol=1e-9)
        except ValueError:
            continue
        if reduced_B.shape[0] < dimension + 1:
            continue
        translation = jax.random.normal(k_trans, (dimension,), dtype=jnp.float64) * translation_scale
        translated_c = jnp.asarray(reduced_c) + jnp.asarray(reduced_B) @ translation
        poly_name = name or f"random-{dimension}d-{attempt}"
        poly_description = description or (
            f"Random half-space polytope with {reduced_B.shape[0]} facets in dimension {dimension}."
        )
        return Polytope(
            name=poly_name,
            B=jnp.asarray(reduced_B, dtype=jnp.float64),
            c=jnp.asarray(translated_c, dtype=jnp.float64),
            description=poly_description,
        )
    raise RuntimeError("Failed to generate a bounded random polytope.")


def random_transformations(
    polytope: Polytope,
    *,
    key: Array,
    count: int,
    scale_range: tuple[float, float] = (0.6, 1.4),
    translation_scale: float = 0.3,
    shear_scale: float = 0.25,
) -> list[Polytope]:
    results: list[Polytope] = []
    k = key
    for _ in range(count):
        subkey, k = jax.random.split(k)
        matrix, translation = random_affine_map(
            polytope.dimension,
            key=subkey,
            scale_range=scale_range,
            shear_scale=shear_scale,
            translation_scale=translation_scale,
        )
        transformed = affine_transform(
            polytope,
            matrix,
            translation=translation,
            name=f"{polytope.name}-random",
            description=f"Random affine perturbation of {polytope.name}",
        )
        results.append(transformed)
    return results


def catalog() -> tuple[Polytope, ...]:
    simplex4 = simplex_with_uniform_weights(4, name="simplex-4d")
    truncated = truncated_simplex_four_dim()
    simplex6 = simplex_with_uniform_weights(6, name="simplex-6d")
    cube4 = hypercube(4, name="hypercube-4d")
    cross4 = cross_polytope(4, name="cross-polytope-4d")
    hexagon_product = regular_polygon_product(
        6,
        6,
        rotation=math.pi / 6,
        name="hexagon-product-rot30",
        description="Product of two hexagons; features twelve facets in dimension four.",
    )
    simplex_product = cartesian_product(
        simplex_with_uniform_weights(2), simplex_with_uniform_weights(2)
    )
    counterexample = viterbo_counterexample()
    return (
        simplex4,
        truncated,
        simplex6,
        cube4,
        cross4,
        hexagon_product,
        simplex_product,
        counterexample,
    )


__all__ = [
    # Data structures
    "Polytope",
    "NormalCone",
    "PolytopeCombinatorics",
    # Half-space utilities
    "enumerate_vertices",
    "remove_redundant_facets",
    "vertices_from_halfspaces",
    "halfspaces_from_vertices",
    # Combinatorics
    "polytope_combinatorics",
    "polytope_fingerprint",
    "clear_polytope_cache",
    # Volume helpers
    "polytope_volume_reference",
    "polytope_volume_fast",
    # Families and transforms
    "hypercube",
    "cross_polytope",
    "regular_polygon_product",
    "cartesian_product",
    "simplex_with_uniform_weights",
    "truncated_simplex_four_dim",
    "viterbo_counterexample",
    "random_affine_map",
    "random_polytope",
    "random_transformations",
    "catalog",
]

