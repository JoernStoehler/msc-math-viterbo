"""Canonical polytope families and affine transforms (datasets layer)."""

from __future__ import annotations

import math
from itertools import product
from typing import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.datasets.builders import build_from_halfspaces
from viterbo.datasets.types import Polytope, PolytopeMetadata, PolytopeRecord


def annotate_polytope(
    geometry: Polytope,
    *,
    slug: str,
    description: str = "",
    reference_capacity: float | None = None,
) -> PolytopeRecord:
    """Attach metadata to a `Polytope` and return a `PolytopeRecord`."""
    metadata = PolytopeMetadata(
        slug=slug,
        description=description,
        reference_capacity=reference_capacity,
    )
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def cartesian_product(
    first: PolytopeRecord,
    second: PolytopeRecord,
    *,
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Return the Cartesian product of two polytopes as a new record."""
    geometry_first = first.geometry
    geometry_second = second.geometry
    B1, c1 = geometry_first.halfspace_data()
    B2, c2 = geometry_second.halfspace_data()
    upper = jnp.hstack((B1, jnp.zeros((B1.shape[0], B2.shape[1]))))
    lower = jnp.hstack((jnp.zeros((B2.shape[0], B1.shape[1])), B2))
    normals = jnp.vstack((upper, lower))
    offsets = jnp.concatenate((c1, c2))
    geometry = build_from_halfspaces(normals, offsets)
    base_slug = slug or f"{first.metadata.slug}x{second.metadata.slug}"
    base_description = description or (
        "Cartesian product constructed from "
        f"{first.metadata.slug} (dim {geometry_first.dimension}) and {second.metadata.slug}"
        f" (dim {geometry_second.dimension})."
    )
    metadata = PolytopeMetadata(slug=base_slug, description=base_description)
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def affine_transform(
    record: PolytopeRecord,
    matrix: Float[Array, " dimension dimension"],
    *,
    translation: Float[Array, " dimension"] | None = None,
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Apply an affine transform to a polytope record."""
    polytope = record.geometry
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
    normals, offsets = polytope.halfspace_data()
    B_transformed = normals @ matrix_inv
    c_transformed = offsets + B_transformed @ translation_vec
    geometry = build_from_halfspaces(B_transformed, c_transformed)
    result_slug = slug or f"{record.metadata.slug}-affine"
    result_description = description or (
        f"Affine image of {record.metadata.slug} via matrix with det {float(det):.3f}."
    )
    metadata = PolytopeMetadata(
        slug=result_slug,
        description=result_description,
        reference_capacity=record.metadata.reference_capacity,
    )
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def translate_polytope(
    polytope: PolytopeRecord,
    translation: Float[Array, " dimension"],
    *,
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Translate a polytope by a vector in-place, returning a new record."""
    return affine_transform(
        polytope,
        jnp.eye(polytope.geometry.dimension),
        translation=translation,
        slug=slug or f"{polytope.metadata.slug}-translated",
        description=description or f"Translation of {polytope.metadata.slug}.",
    )


def mirror_polytope(
    polytope: PolytopeRecord,
    axes: Sequence[bool],
    *,
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Reflect a polytope across selected coordinate axes."""
    if len(tuple(axes)) != polytope.geometry.dimension:
        raise ValueError("Axis mask must match the polytope dimension.")
    signs = jnp.where(jnp.asarray(tuple(axes), dtype=bool), -1.0, 1.0)
    matrix = jnp.diag(signs)
    return affine_transform(
        polytope,
        matrix,
        translation=jnp.zeros(polytope.geometry.dimension),
        slug=slug or f"{polytope.metadata.slug}-mirrored",
        description=description or "Coordinate reflection of the base polytope.",
    )


def rotate_polytope(
    polytope: PolytopeRecord,
    *,
    plane: tuple[int, int],
    angle: float,
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Rotate a polytope within a coordinate plane by `angle` (radians)."""
    i, j = plane
    dimension = polytope.geometry.dimension
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
        slug=slug or f"{polytope.metadata.slug}-rot",
        description=description
        or f"Rotation of {polytope.metadata.slug} in plane {(i, j)} by {angle} rad.",
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
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Return the 4D polytope that is the product of two regular polygons."""
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
    default_slug = slug or (
        f"{sides_first}gonx{sides_second}gon-rot{int(round(math.degrees(rotation)))}"
    )
    default_description = description or (
        "Product of two regular polygons, yielding a 4D polytope with"
        f" {normals_first.shape[0] + rotated_second.shape[0]} facets."
    )
    geometry = build_from_halfspaces(
        jnp.asarray(B_np, dtype=jnp.float64),
        jnp.asarray(c_np, dtype=jnp.float64),
    )
    metadata = PolytopeMetadata(slug=default_slug, description=default_description)
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def cross_polytope(
    dimension: int,
    *,
    radius: float = 1.0,
    slug: str | None = None,
) -> PolytopeRecord:
    """Return the cross-polytope (L1 ball) of given dimension."""
    if dimension < 2:
        raise ValueError("Dimension must be at least two.")
    normals = jnp.asarray(list(product((-1.0, 1.0), repeat=dimension)), dtype=jnp.float64)
    c = jnp.full(normals.shape[0], float(radius))
    description = "Centrally symmetric cross-polytope with L1 ball geometry."
    geometry = build_from_halfspaces(normals, c)
    metadata = PolytopeMetadata(
        slug=slug or f"cross-polytope-{dimension}d", description=description
    )
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def hypercube(
    dimension: int,
    *,
    radius: float = 1.0,
    slug: str | None = None,
) -> PolytopeRecord:
    """Return the axis-aligned hypercube of given radius and dimension."""
    if dimension < 2:
        raise ValueError("Dimension must be at least two.")
    identity = jnp.eye(dimension)
    B_matrix = jnp.vstack((identity, -identity))
    c = jnp.full(2 * dimension, float(radius))
    description = "Hypercube aligned with the coordinate axes."
    geometry = build_from_halfspaces(B_matrix, c)
    metadata = PolytopeMetadata(slug=slug or f"hypercube-{dimension}d", description=description)
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def simplex_with_uniform_weights(
    dimension: int,
    *,
    last_offset: float | None = None,
    slug: str | None = None,
) -> PolytopeRecord:
    """Return a simplex with uniform coordinate facets and a balancing facet."""
    if dimension < 2:
        raise ValueError("Dimension must be at least two.")
    B_matrix = jnp.eye(dimension)
    extra = -jnp.ones((1, dimension))
    B_matrix = jnp.vstack((B_matrix, extra))
    offsets = jnp.ones(dimension + 1)
    if last_offset is None:
        last_offset = dimension / 2
    offsets = offsets.at[-1].set(float(last_offset))
    polytope_slug = slug or f"uniform-simplex-{dimension}d"
    reference_capacity = None
    if dimension == 4 and math.isclose(last_offset, dimension / 2):
        reference_capacity = 9.0
    description = (
        "4D simplex with symmetric coordinates; used as a canonical regression example"
        if dimension == 4
        else "Simplex with uniform coordinate facets and a balancing facet."
    )
    geometry = build_from_halfspaces(B_matrix, offsets)
    metadata = PolytopeMetadata(
        slug=polytope_slug,
        description=description,
        reference_capacity=reference_capacity,
    )
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def catalog() -> tuple[PolytopeRecord, ...]:
    """Return the canonical catalog of polytopes used in tests and demos."""
    simplex4 = simplex_with_uniform_weights(4, slug="simplex-4d")
    truncated = truncated_simplex_four_dim()
    simplex6 = simplex_with_uniform_weights(6, slug="simplex-6d")
    cube4 = hypercube(4, slug="hypercube-4d")
    cross4 = cross_polytope(4, slug="cross-polytope-4d")
    hexagon_product = regular_polygon_product(
        6,
        6,
        rotation=math.pi / 6,
        slug="hexagon-product-rot30",
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


def truncated_simplex_four_dim() -> PolytopeRecord:
    """Return a hand-tuned truncated 4D simplex with known reference capacity."""
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

    from viterbo.math.symplectic import standard_symplectic_matrix as _J

    def _haim_kislev_action(B: jnp.ndarray, c_vec: jnp.ndarray) -> float:
        order = (2, 0, 4, 3, 1)
        subset = (0, 1, 2, 3, 4)
        m = len(subset)
        system = jnp.zeros((m, m), dtype=jnp.float64)
        idx = jnp.array(subset, dtype=jnp.int32)
        system = system.at[0, :].set(c_vec[idx])
        B_subset = B[idx, :]
        system = system.at[1:, :].set(B_subset.T)
        rhs = jnp.zeros((m,), dtype=jnp.float64)
        rhs = rhs.at[0].set(1.0)
        beta = jnp.linalg.solve(system, rhs)
        J = _J(4)
        W = B_subset @ J @ B_subset.T
        value = 0.0
        for i in range(1, m):
            for j in range(i):
                value += float(beta[order[i]] * beta[order[j]] * W[order[i], order[j]])
        return float(0.5 / value)

    geometry = build_from_halfspaces(B_matrix, c)
    ref = _haim_kislev_action(B_matrix, c)
    metadata = PolytopeMetadata(
        slug="truncated-simplex-4d",
        description="Truncated 4D simplex with manually tuned facet.",
        reference_capacity=ref,
    )
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def viterbo_counterexample(radius: float = 1.0) -> PolytopeRecord:
    """Return the 4D Viterbo counterexample polytope record."""
    normals = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    offsets = jnp.array([radius, radius, radius, radius, 2.0 * radius, 1.2 * radius])
    geometry = build_from_halfspaces(normals, offsets)
    metadata = PolytopeMetadata(
        slug="viterbo-counterexample",
        description="4D counterexample polytope used for regression tests.",
        reference_capacity=None,
    )
    return PolytopeRecord(geometry=geometry, metadata=metadata)


def random_affine_map(
    dimension: int,
    *,
    key: Array,
    scale_range: tuple[float, float] = (0.6, 1.4),
    translation_scale: float = 0.3,
    shear_scale: float = 0.25,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random invertible affine map (matrix, translation)."""
    low, high = scale_range
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    scales = jax.random.uniform(subkey1, (dimension,), minval=low, maxval=high, dtype=jnp.float64)
    shears = jax.random.normal(subkey2, (dimension, dimension), dtype=jnp.float64) * shear_scale
    shears = jnp.triu(shears, k=1)
    matrix = jnp.diag(scales) + shears
    translation = jax.random.normal(subkey3, (dimension,), dtype=jnp.float64) * translation_scale
    return matrix, translation


def random_polytope(
    *,
    dimension: int,
    facets: int,
    key: Array,
    low: float = 0.5,
    high: float = 1.5,
    translation_scale: float = 0.1,
    slug: str | None = None,
    description: str | None = None,
) -> PolytopeRecord:
    """Sample a bounded random half-space polytope record in `dimension`."""
    identity = jnp.eye(dimension)
    k_state = key
    for attempt in range(50):
        k_state, k_normals, k_offsets, k_trans = jax.random.split(k_state, 4)
        normals = jax.random.normal(k_normals, (facets, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.clip(norms, a_min=1e-12, a_max=None)
        offsets = jax.random.uniform(
            k_offsets, (facets,), minval=low, maxval=high, dtype=jnp.float64
        )
        normals = jnp.vstack((normals, identity, -identity))
        offsets = jnp.concatenate((offsets, jnp.full(2 * dimension, float(high))))
        try:
            from viterbo.math.geometry import remove_redundant_facets

            reduced_B, reduced_c = remove_redundant_facets(normals, offsets, atol=1e-9)
        except ValueError:
            continue
        if reduced_B.shape[0] < dimension + 1:
            continue
        translation = (
            jax.random.normal(k_trans, (dimension,), dtype=jnp.float64) * translation_scale
        )
        translated_c = jnp.asarray(reduced_c) + jnp.asarray(reduced_B) @ translation
        geometry = build_from_halfspaces(
            jnp.asarray(reduced_B, dtype=jnp.float64),
            jnp.asarray(translated_c, dtype=jnp.float64),
        )
        metadata = PolytopeMetadata(
            slug=slug or f"random-{dimension}d-{attempt}",
            description=description
            or (
                f"Random half-space polytope with {reduced_B.shape[0]} facets in dimension {dimension}."
            ),
            reference_capacity=None,
        )
        return PolytopeRecord(geometry=geometry, metadata=metadata)
    raise RuntimeError("Failed to generate a bounded random polytope.")


def random_transformations(
    polytope: PolytopeRecord,
    *,
    key: Array,
    count: int,
    scale_range: tuple[float, float] = (0.6, 1.4),
    translation_scale: float = 0.3,
    shear_scale: float = 0.25,
) -> list[PolytopeRecord]:
    """Generate `count` random affine images of a polytope record."""
    results: list[PolytopeRecord] = []
    k = key
    for _ in range(count):
        subkey, k = jax.random.split(k)
        matrix, translation = random_affine_map(
            polytope.geometry.dimension,
            key=subkey,
            scale_range=scale_range,
            shear_scale=shear_scale,
            translation_scale=translation_scale,
        )
        transformed = affine_transform(
            polytope,
            matrix,
            translation=translation,
            slug=f"{polytope.metadata.slug}-random",
            description=f"Random affine perturbation of {polytope.metadata.slug}",
        )
        results.append(transformed)
    return results


__all__ = [
    "annotate_polytope",
    "cartesian_product",
    "affine_transform",
    "translate_polytope",
    "mirror_polytope",
    "rotate_polytope",
    "regular_polygon_product",
    "cross_polytope",
    "hypercube",
    "simplex_with_uniform_weights",
    "truncated_simplex_four_dim",
    "viterbo_counterexample",
    "random_affine_map",
    "random_polytope",
    "random_transformations",
    "catalog",
]
