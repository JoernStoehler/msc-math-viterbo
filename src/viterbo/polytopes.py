"""Canonical polytope families and deterministic transformations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Final, Sequence

import numpy as np
from jaxtyping import Float

from .core import standard_symplectic_matrix
from .halfspaces import enumerate_vertices, remove_redundant_facets

try:  # Optional dependency used only when available.
    from scipy.spatial import ConvexHull
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without SciPy.
    ConvexHull = None

_DIMENSION_AXIS: Final[str] = "dimension"
_FACET_AXIS: Final[str] = "num_facets"
_FACET_MATRIX_AXES: Final[str] = f"{_FACET_AXIS} {_DIMENSION_AXIS}"
_SQUARE_MATRIX_AXES: Final[str] = f"{_DIMENSION_AXIS} {_DIMENSION_AXIS}"
_VERTEX_MATRIX_AXES: Final[str] = "num_vertices dimension"

_POLYTOPE_CACHE: dict[tuple[str, float], PolytopeCombinatorics] = {}

__all__ = [
    "Polytope",
    "NormalCone",
    "PolytopeCombinatorics",
    "affine_transform",
    "cartesian_product",
    "catalog",
    "cross_polytope",
    "haim_kislev_action",
    "hypercube",
    "mirror_polytope",
    "random_affine_map",
    "random_polytope",
    "random_transformations",
    "regular_polygon_product",
    "vertices_from_halfspaces",
    "halfspaces_from_vertices",
    "polytope_fingerprint",
    "polytope_combinatorics",
    "rotate_polytope",
    "simplex_with_uniform_weights",
    "translate_polytope",
    "truncated_simplex_four_dim",
    "viterbo_counterexample",
]


@dataclass(frozen=True)
class NormalCone:
    """Normal cone data attached to a polytope vertex."""

    vertex: Float[np.ndarray, _DIMENSION_AXIS]
    active_facets: tuple[int, ...]
    normals: Float[np.ndarray, "num_active dimension"]

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


def vertices_from_halfspaces(
    B: Float[np.ndarray, _FACET_MATRIX_AXES],
    c: Float[np.ndarray, _FACET_AXIS],
    *,
    atol: float = 1e-9,
) -> Float[np.ndarray, _VERTEX_MATRIX_AXES]:
    """Enumerate the vertices of a polytope described by ``Bx <= c``."""
    return enumerate_vertices(B, c, atol=atol)


def halfspaces_from_vertices(
    vertices: Float[np.ndarray, _VERTEX_MATRIX_AXES],
    *,
    qhull_options: str | None = None,
) -> tuple[
    Float[np.ndarray, _FACET_MATRIX_AXES],
    Float[np.ndarray, _FACET_AXIS],
]:
    """Return a half-space description from a vertex set using Qhull."""
    if ConvexHull is None:  # pragma: no cover - SciPy is an optional dependency.
        msg = "scipy is required to convert vertices to half-space form."
        raise ModuleNotFoundError(msg)

    hull = ConvexHull(np.asarray(vertices, dtype=float), qhull_options=qhull_options)
    # Qhull stores inequalities as <normal, offset> with normal pointing outward
    # and the inequality ``normal @ x + offset <= 0``. We flip the sign so that
    # rows match the ``Bx <= c`` convention used throughout the project.
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]
    B = normals
    c = -offsets
    return remove_redundant_facets(B, c)


def polytope_combinatorics(
    polytope: Polytope,
    *,
    atol: float = 1e-9,
    use_cache: bool = True,
) -> PolytopeCombinatorics:
    """Return cached combinatorial data for ``polytope``."""
    key = (polytope_fingerprint(polytope), float(np.round(atol, decimals=12)))
    if use_cache and key in _POLYTOPE_CACHE:
        return _POLYTOPE_CACHE[key]

    B, c = polytope.halfspace_data()
    vertices = enumerate_vertices(B, c, atol=atol)
    adjacency = np.zeros((polytope.facets, polytope.facets), dtype=bool)
    normal_cones: list[NormalCone] = []

    for vertex in vertices:
        residuals = B @ vertex - c
        active = np.where(np.abs(residuals) <= atol)[0]
        if active.size == 0:
            continue

        for first, second in combinations(active, 2):
            adjacency[first, second] = True
            adjacency[second, first] = True

        normals = B[active, :]
        normal_cones.append(
            NormalCone(
                vertex=vertex,
                active_facets=tuple(int(index) for index in active),
                normals=normals,
            )
        )

    np.fill_diagonal(adjacency, False)
    combinatorics = PolytopeCombinatorics(
        vertices=vertices,
        facet_adjacency=adjacency,
        normal_cones=tuple(normal_cones),
    )

    if use_cache:
        _POLYTOPE_CACHE[key] = combinatorics

    return combinatorics


@dataclass(frozen=True)
class Polytope:
    """Immutable container describing a convex polytope via half-space data."""

    name: str
    B: Float[np.ndarray, _FACET_MATRIX_AXES]  # shape: (num_facets, dimension)
    c: Float[np.ndarray, _FACET_AXIS]  # shape: (num_facets,)
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


def cartesian_product(
    first: Polytope,
    second: Polytope,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    r"""
    Return the Cartesian product of two polytopes.

    The inputs ``(B_1, c_1)`` and ``(B_2, c_2)`` are combined into a block-diagonal
    system representing the product polytope in ``\mathbb{R}^{d_1 + d_2}``.
    """
    B1, c1 = first.halfspace_data()
    B2, c2 = second.halfspace_data()

    upper = np.hstack((B1, np.zeros((B1.shape[0], B2.shape[1]))))
    lower = np.hstack((np.zeros((B2.shape[0], B1.shape[1])), B2))
    B = np.vstack((upper, lower))
    c = np.concatenate((c1, c2))

    product_name = name or f"{first.name}x{second.name}"
    product_description = description or (
        "Cartesian product constructed from "
        f"{first.name} (dim {first.dimension}) and {second.name} (dim {second.dimension})."
    )
    return Polytope(name=product_name, B=B, c=c, description=product_description)


def affine_transform(
    polytope: Polytope,
    matrix: Float[np.ndarray, _SQUARE_MATRIX_AXES],
    *,
    translation: Float[np.ndarray, _DIMENSION_AXIS] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    r"""
    Apply an invertible affine transformation to ``polytope``.

    For ``y = A x + t`` with invertible ``A``, the transformed inequality system is
    ``B' y \le c'`` where ``B' = B A^{-1}`` and ``c' = c + B' t``.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (polytope.dimension, polytope.dimension):
        msg = "Linear transform must match the ambient dimension."
        raise ValueError(msg)

    try:
        matrix_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        msg = "Affine transform requires an invertible matrix."
        raise ValueError(msg) from exc

    translation_vec = (
        np.zeros(polytope.dimension)
        if translation is None
        else np.asarray(translation, dtype=float)
    )
    if translation_vec.shape != (polytope.dimension,):
        msg = "Translation vector must match the ambient dimension."
        raise ValueError(msg)

    B_transformed = polytope.B @ matrix_inv
    c_transformed = polytope.c + B_transformed @ translation_vec

    return Polytope(
        name=name or f"{polytope.name}-affine",
        B=B_transformed,
        c=c_transformed,
        description=description
        or (f"Affine image of {polytope.name} via matrix with det {np.linalg.det(matrix):.3f}."),
        reference_capacity=None,
    )


def translate_polytope(
    polytope: Polytope,
    translation: Float[np.ndarray, _DIMENSION_AXIS],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    """Return a translated copy of ``polytope``."""
    translation_vec = np.asarray(translation, dtype=float)
    if translation_vec.shape != (polytope.dimension,):
        msg = "Translation vector must match the ambient dimension."
        raise ValueError(msg)

    return affine_transform(
        polytope,
        np.eye(polytope.dimension),
        translation=translation_vec,
        name=name or f"{polytope.name}-translated",
        description=description or f"Translation of {polytope.name} by {translation_vec}.",
    )


def mirror_polytope(
    polytope: Polytope,
    axes: Sequence[bool],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    """Reflect ``polytope`` across the coordinate axes toggled by ``axes``."""
    if len(tuple(axes)) != polytope.dimension:
        msg = "Axis mask must match the polytope dimension."
        raise ValueError(msg)

    signs = np.where(np.asarray(tuple(axes), dtype=bool), -1.0, 1.0)
    matrix = np.diag(signs)
    return affine_transform(
        polytope,
        matrix,
        translation=np.zeros(polytope.dimension),
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
    """Rotate ``polytope`` within the two-dimensional ``plane`` by ``angle`` radians."""
    i, j = plane
    dimension = polytope.dimension
    if not (0 <= i < dimension and 0 <= j < dimension) or i == j:
        msg = "Rotation plane indices must be distinct and within range."
        raise ValueError(msg)

    rotation = np.eye(dimension)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation[i, i] = cosine
    rotation[j, j] = cosine
    rotation[i, j] = -sine
    rotation[j, i] = sine

    return affine_transform(
        polytope,
        rotation,
        translation=np.zeros(dimension),
        name=name or f"{polytope.name}-rot",
        description=description or f"Rotation of {polytope.name} in plane {(i, j)} by {angle} rad.",
    )


def random_affine_map(
    dimension: int,
    *,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.6, 1.4),
    shear_scale: float = 0.25,
    translation_scale: float = 0.3,
) -> tuple[
    Float[np.ndarray, _SQUARE_MATRIX_AXES],
    Float[np.ndarray, _DIMENSION_AXIS],
]:
    """Sample a well-conditioned random affine map for deterministic experiments."""
    lower, upper = scale_range
    if lower <= 0 or upper <= 0:
        msg = "Scale factors must be positive."
        raise ValueError(msg)

    for _ in range(32):
        q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        scales = rng.uniform(lower, upper, size=dimension)
        shear = np.eye(dimension) + rng.normal(scale=shear_scale, size=(dimension, dimension))
        matrix = q @ np.diag(scales) @ shear
        try:
            np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue
        translation = rng.normal(scale=translation_scale, size=dimension)
        return matrix, translation

    msg = "Failed to sample an invertible affine map."
    raise RuntimeError(msg)


def random_polytope(
    dimension: int,
    *,
    rng: np.random.Generator,
    facets: int | None = None,
    offset_range: tuple[float, float] = (0.5, 1.5),
    translation_scale: float = 0.2,
    name: str | None = None,
    description: str | None = None,
    max_attempts: int = 64,
) -> Polytope:
    """Sample a bounded random polytope with redundant facets removed."""
    if dimension <= 0:
        msg = "Dimension must be positive."
        raise ValueError(msg)

    low, high = offset_range
    if low <= 0 or high <= 0 or high <= low:
        msg = "Offsets must satisfy 0 < low < high."
        raise ValueError(msg)

    if facets is None:
        facets = max(dimension + 1, 4 * dimension)
    if facets < dimension + 1:
        msg = "At least dimension + 1 facets are required."
        raise ValueError(msg)

    identity = np.eye(dimension)
    for attempt in range(max_attempts):
        normals = rng.normal(size=(facets, dimension))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(norms, a_min=1e-12, a_max=None)
        offsets = rng.uniform(low, high, size=facets)

        normals = np.vstack((normals, identity, -identity))
        offsets = np.concatenate((offsets, np.full(2 * dimension, high)))

        try:
            reduced_B, reduced_c = remove_redundant_facets(normals, offsets, atol=1e-9)
        except ValueError:
            continue

        if reduced_B.shape[0] < dimension + 1:
            continue

        translation = rng.normal(scale=translation_scale, size=dimension)
        translated_c = reduced_c + reduced_B @ translation

        poly_name = name or f"random-{dimension}d-{attempt}"
        poly_description = description or (
            f"Random half-space polytope with {reduced_B.shape[0]} facets in dimension {dimension}."
        )

        return Polytope(
            name=poly_name,
            B=reduced_B,
            c=translated_c,
            description=poly_description,
        )

    msg = "Failed to generate a bounded random polytope."
    raise RuntimeError(msg)


def haim_kislev_action(
    B: np.ndarray,
    c: np.ndarray,
    *,
    subset: Sequence[int],
    order: Sequence[int],
) -> float:
    """Evaluate the Haim–Kislev action for a facet subset and total order."""
    matrix = np.asarray(B, dtype=float)
    offsets = np.asarray(c, dtype=float)
    if matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        msg = "Vector c must match the number of facets."
        raise ValueError(msg)

    dimension = matrix.shape[1]
    J = standard_symplectic_matrix(dimension)

    rows = np.asarray(tuple(subset), dtype=int)
    B_subset = matrix[rows]
    c_subset = offsets[rows]
    m = len(rows)

    order_tuple = tuple(order)
    if len(order_tuple) != m:
        msg = "Facet order must include each subset index exactly once."
        raise ValueError(msg)
    if not all(isinstance(idx, (int, np.integer)) for idx in order_tuple):
        msg = "Facet order must contain integer indices."
        raise ValueError(msg)

    order_indices = np.asarray(order_tuple, dtype=int)
    if not np.array_equal(np.sort(order_indices), np.arange(m)):
        msg = "Facet order must be a permutation of range(m)."
        raise ValueError(msg)

    system = np.zeros((m, m))
    system[0, :] = c_subset
    system[1:, :] = B_subset.T

    rhs = np.zeros(m)
    rhs[0] = 1.0
    beta = np.linalg.solve(system, rhs)

    symplectic_products = (B_subset @ J) @ B_subset.T

    total = 0.0
    for i in range(1, m):
        idx_i = order[i]
        weight_i = beta[idx_i]
        if weight_i <= 0.0:
            continue
        row = symplectic_products[idx_i]
        for j in range(i):
            idx_j = order[j]
            weight_j = beta[idx_j]
            if weight_j <= 0.0:
                continue
            total += weight_i * weight_j * row[idx_j]

    if total <= 0.0:
        msg = "Facet ordering yielded a non-positive action."
        raise ValueError(msg)

    return 0.5 / total


def simplex_with_uniform_weights(
    dimension: int,
    *,
    last_offset: float | None = None,
    name: str | None = None,
) -> Polytope:
    """Return the simplex used for regression tests with uniform weights."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

    B = np.eye(dimension)
    extra = -np.ones((1, dimension))
    B = np.vstack((B, extra))

    offsets = np.ones(dimension + 1)
    if last_offset is None:
        last_offset = dimension / 2
    offsets[-1] = float(last_offset)

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
        B=B,
        c=offsets,
        description=description,
        reference_capacity=reference_capacity,
    )


def truncated_simplex_four_dim() -> Polytope:
    """Return the 4D simplex truncated by an additional slanted facet."""
    B = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    c = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.2])
    reference_capacity = haim_kislev_action(
        B,
        c,
        subset=(0, 1, 2, 3, 4),
        order=(2, 0, 4, 3, 1),
    )
    description = "Simplex-like model with an extra facet; preserves the optimal Reeb action"
    return Polytope(
        name="truncated-simplex-4d",
        B=B,
        c=c,
        description=description,
        reference_capacity=reference_capacity,
    )


def cross_polytope(
    dimension: int,
    *,
    radius: float = 1.0,
    name: str | None = None,
) -> Polytope:
    """Return the centrally symmetric cross-polytope of the given radius."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

    normals = np.array(list(product((-1.0, 1.0), repeat=dimension)))
    c = np.full(normals.shape[0], float(radius))
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
    """Return the axis-aligned hypercube of side length ``2 * radius``."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

    identity = np.eye(dimension)
    B = np.vstack((identity, -identity))
    c = np.full(2 * dimension, float(radius))
    description = "Hypercube aligned with the coordinate axes."
    return Polytope(
        name=name or f"hypercube-{dimension}d",
        B=B,
        c=c,
        description=description,
    )


def _regular_polygon_normals(sides: int) -> np.ndarray:
    if sides < 3:
        msg = "A polygon requires at least three sides."
        raise ValueError(msg)

    angles = 2 * np.pi * (np.arange(sides) / sides)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    return normals


def _rotation_matrix(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


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
    r"""Return the direct product of two regular polygons in ``\mathbb{R}^4``."""
    normals_first = _regular_polygon_normals(sides_first)
    normals_second = _regular_polygon_normals(sides_second)
    if radius_second is None:
        radius_second = radius_first

    rotation_matrix = _rotation_matrix(rotation)
    rotated_second = normals_second @ rotation_matrix.T

    zero_block = np.zeros((normals_first.shape[0], 2))
    B_upper = np.hstack((normals_first, zero_block))
    B_lower = np.hstack((np.zeros((rotated_second.shape[0], 2)), rotated_second))
    B = np.vstack((B_upper, B_lower))
    c = np.concatenate(
        (
            np.full(normals_first.shape[0], float(radius_first)),
            np.full(rotated_second.shape[0], float(radius_second)),
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
        B=B,
        c=c,
        description=default_description,
    )


def viterbo_counterexample(radius: float = 1.0) -> Polytope:
    """Return the Chaidez–Hutchings counterexample to Viterbo's conjecture."""
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


def random_transformations(
    polytope: Polytope,
    *,
    rng: np.random.Generator,
    count: int,
    scale_range: tuple[float, float] = (0.6, 1.4),
    translation_scale: float = 0.3,
    shear_scale: float = 0.25,
) -> list[
    tuple[
        Float[np.ndarray, _FACET_MATRIX_AXES],
        Float[np.ndarray, _FACET_AXIS],
    ]
]:
    """Generate random linear transformations and translations of ``polytope``."""
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(count):
        matrix, translation = random_affine_map(
            polytope.dimension,
            rng=rng,
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
        results.append(transformed.halfspace_data())
    return results


def catalog() -> tuple[Polytope, ...]:
    """Return a curated tuple of polytopes used for regression and profiling."""
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
