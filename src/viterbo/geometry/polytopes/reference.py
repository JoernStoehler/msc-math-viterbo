"""Canonical polytope families and deterministic transformations.

Caching policy
--------------
- Combinatorics are cached by a fingerprint of ``(B, c)`` and the tolerance.
- Cache is bounded (LRU) with a hard size limit set by
  ``_POLYTOPE_CACHE_MAX_SIZE``.
- Disable caching by setting environment variable ``VITERBO_DISABLE_CACHE=1``
  or by passing ``use_cache=False`` to ``polytope_combinatorics``.
- Invalidate all entries via ``clear_polytope_cache()``.
"""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Sequence

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np
from jaxtyping import Array, Float

from viterbo._wrapped.spatial import convex_hull_equations
from viterbo.geometry.halfspaces import enumerate_vertices, remove_redundant_facets
from viterbo.geometry.polytopes import _shared
from viterbo.symplectic.core import standard_symplectic_matrix

NormalCone = _shared.NormalCone
Polytope = _shared.Polytope
PolytopeCombinatorics = _shared.PolytopeCombinatorics
clear_polytope_cache = _shared.clear_polytope_cache
polytope_fingerprint = _shared.polytope_fingerprint


def vertices_from_halfspaces(
    B: Float[Array, " num_facets dimension"],
    c: Float[Array, " num_facets"],
    *,
    atol: float = 1e-9,
) -> Float[Array, " num_vertices dimension"]:
    """Enumerate the vertices of a polytope described by ``Bx <= c``."""
    return enumerate_vertices(B, c, atol=atol)


def halfspaces_from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
    *,
    qhull_options: str | None = None,
) -> tuple[
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]:
    """Return a half-space description from a vertex set using Qhull."""
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
    """Return cached combinatorial data for ``polytope``.

    Uses an LRU cache with a hard size bound. Caching can be disabled by
    setting the environment variable ``VITERBO_DISABLE_CACHE=1`` or by
    passing ``use_cache=False``.
    """
    key = _shared.polytope_cache_key(polytope, atol)
    if _shared.cache_enabled(use_cache):
        cached = _shared.cache_lookup(key)
        if cached is not None:
            return cached

    B, c = polytope.halfspace_data()
    vertices = enumerate_vertices(B, c, atol=atol)
    combinatorics = _shared.build_combinatorics(B, c, vertices, atol=atol)

    if _shared.cache_enabled(use_cache):
        _shared.cache_store(key, combinatorics)

    return combinatorics


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
    matrix_inverse: Float[Array, " dimension dimension"] | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    r"""
    Apply an invertible affine transformation to ``polytope``.

    For ``y = A x + t`` with invertible ``A``, the transformed inequality system is
    ``B' y \le c'`` where ``B' = B A^{-1}`` and ``c' = c + B' t``.

    Args:
      polytope: Polytope to transform. Must expose ``dimension`` matching ``matrix``.
      matrix: Linear component ``A``.
      translation: Optional translation ``t``.
      matrix_inverse: Optional precomputed inverse of ``matrix``. When provided
        the inverse is validated and reused instead of recomputing it.
      name: Optional override for the transformed polytope's name. Defaults to
        ``"{polytope.name}-affine"`` when not provided.
      description: Optional textual description for the transformed polytope.
    """
    matrix = jnp.asarray(matrix, dtype=jnp.float64)
    if matrix.shape != (polytope.dimension, polytope.dimension):
        msg = "Linear transform must match the ambient dimension."
        raise ValueError(msg)

    if matrix_inverse is None:
        det = jnp.linalg.det(matrix)
        if jnp.isclose(det, 0.0):
            msg = "Affine transform requires an invertible matrix."
            raise ValueError(msg)
        matrix_inv = jnp.linalg.inv(matrix)
    else:
        matrix_inv = jnp.asarray(matrix_inverse, dtype=jnp.float64)
        if matrix_inv.shape != (polytope.dimension, polytope.dimension):
            msg = "Matrix inverse must match the ambient dimension."
            raise ValueError(msg)

        if not bool(
            jnp.allclose(matrix @ matrix_inv, jnp.eye(polytope.dimension), atol=1e-9).item()
        ):
            msg = "Provided matrix_inverse is not a valid inverse of matrix."
            raise ValueError(msg)

    translation_vec = (
        jnp.zeros(polytope.dimension)
        if translation is None
        else jnp.asarray(translation, dtype=jnp.float64)
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
        or (
            f"Affine image of {polytope.name} via matrix with det {float(jnp.linalg.det(matrix)):.3f}."
        ),
        reference_capacity=None,
    )


def translate_polytope(
    polytope: Polytope,
    translation: Float[Array, " dimension"] | Float[np.ndarray, " dimension"],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    """Return a translated copy of ``polytope``."""
    translation_vec = jnp.asarray(translation, dtype=jnp.float64)
    if translation_vec.shape != (polytope.dimension,):
        msg = "Translation vector must match the ambient dimension."
        raise ValueError(msg)

    return affine_transform(
        polytope,
        jnp.eye(polytope.dimension),
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
    """Rotate ``polytope`` within the two-dimensional ``plane`` by ``angle`` radians."""
    i, j = plane
    dimension = polytope.dimension
    if not (0 <= i < dimension and 0 <= j < dimension) or i == j:
        msg = "Rotation plane indices must be distinct and within range."
        raise ValueError(msg)

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


def random_affine_map(
    dimension: int,
    *,
    key: Array,
    scale_range: tuple[float, float] = (0.6, 1.4),
    shear_scale: float = 0.25,
    translation_scale: float = 0.3,
) -> tuple[
    Float[Array, " dimension dimension"],
    Float[Array, " dimension"],
]:
    """Sample a well-conditioned random affine map for deterministic experiments."""
    lower, upper = scale_range
    if lower <= 0 or upper <= 0:
        msg = "Scale factors must be positive."
        raise ValueError(msg)

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

    msg = "Failed to sample an invertible affine map."
    raise RuntimeError(msg)


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

    identity = jnp.eye(dimension)
    k = key
    for attempt in range(max_attempts):
        k, k_normals, k_offsets, k_trans = jax.random.split(
            jax.random.PRNGKey(jax.random.randint(k, (), 0, 2**31 - 1, dtype=jnp.uint32).item()), 4
        )
        normals = jax.random.normal(k_normals, (facets, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.clip(norms, a_min=1e-12, a_max=None)
        offsets = jax.random.uniform(
            k_offsets, (facets,), minval=low, maxval=high, dtype=jnp.float64
        )

        normals = jnp.vstack((normals, identity, -identity))
        offsets = jnp.concatenate((offsets, jnp.full(2 * dimension, float(high))))

        try:
            reduced_B, reduced_c = remove_redundant_facets(normals, offsets, atol=1e-9)
        except ValueError:
            continue

        if reduced_B.shape[0] < dimension + 1:
            continue

        translation = (
            jax.random.normal(k_trans, (dimension,), dtype=jnp.float64) * translation_scale
        )
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

    msg = (
        "Failed to generate a bounded random polytope after "
        f"{max_attempts} attempts (dimension={dimension}, facets={facets}, "
        f"offset_range={offset_range}, translation_scale={translation_scale})."
    )
    raise RuntimeError(msg)


def haim_kislev_action(
    B: Array,
    c: Array,
    *,
    subset: Sequence[int],
    order: Sequence[int],
) -> float:
    """Evaluate the Haim–Kislev action for a facet subset and total order."""
    matrix = jnp.asarray(B, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)
    if matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        msg = "Vector c must match the number of facets."
        raise ValueError(msg)

    dimension = matrix.shape[1]
    J = standard_symplectic_matrix(dimension)

    rows = jnp.asarray(tuple(subset), dtype=int)
    B_subset = matrix[rows]
    c_subset = offsets[rows]
    m = len(rows)

    order_tuple = tuple(order)
    if len(order_tuple) != m:
        msg = "Facet order must include each subset index exactly once."
        raise ValueError(msg)
    # Indices are integers by contract from the function signature; permutation check follows.

    order_indices = jnp.asarray(order_tuple, dtype=int)
    if sorted(list(map(int, order_indices.tolist()))) != list(range(m)):
        msg = "Facet order must be a permutation of range(m)."
        raise ValueError(msg)

    system = jnp.zeros((m, m))
    system = system.at[0, :].set(c_subset)
    system = system.at[1:, :].set(B_subset.T)

    rhs = jnp.zeros(m)
    rhs = rhs.at[0].set(1.0)
    beta = jnp.linalg.solve(system, rhs)

    symplectic_products = (B_subset @ jnp.asarray(J)) @ B_subset.T

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

    return 0.5 / float(total)


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
    """Return the 4D simplex truncated by an additional slanted facet."""
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
    reference_capacity = haim_kislev_action(
        B_matrix,
        c,
        subset=(0, 1, 2, 3, 4),
        order=(2, 0, 4, 3, 1),
    )
    description = "Simplex-like model with an extra facet; preserves the optimal Reeb action"
    return Polytope(
        name="truncated-simplex-4d",
        B=B_matrix,
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
    """Return the axis-aligned hypercube of side length ``2 * radius``."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

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


def _regular_polygon_normals(sides: int) -> jnp.ndarray:
    if sides < 3:
        msg = "A polygon requires at least three sides."
        raise ValueError(msg)

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
    r"""Return the direct product of two regular polygons in ``\mathbb{R}^4``."""
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
    key: Array,
    count: int,
    scale_range: tuple[float, float] = (0.6, 1.4),
    translation_scale: float = 0.3,
    shear_scale: float = 0.25,
) -> list[Polytope]:
    """Generate random linear transformations and translations of ``polytope``."""
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
