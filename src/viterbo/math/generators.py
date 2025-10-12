"""Polytope generators and samplers (pure math layer).

All functions return JAX arrays or tuples of arrays. No dataclasses.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from viterbo._wrapped import spatial as _spatial
from viterbo.math.numerics import GEOMETRY_ABS_TOLERANCE


def from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
) -> tuple[
    Float[Array, " num_vertices dimension"],
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]:
    """Return (hull_vertices, normals, offsets) from an input vertex cloud."""

    verts = jnp.asarray(vertices, dtype=jnp.float64)
    equations = _spatial.convex_hull_equations(verts)
    normals = jnp.asarray(equations[:, :-1], dtype=jnp.float64)
    offsets = jnp.asarray(-equations[:, -1], dtype=jnp.float64)
    hull_indices = jnp.asarray(_spatial.convex_hull_vertices(verts), dtype=jnp.int32)
    hull_vertices = verts[hull_indices]
    return hull_vertices, normals, offsets


def from_halfspaces(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> Float[Array, " num_vertices dimension"]:
    """Return vertices from a half-space description (may be empty)."""

    normals64 = jnp.asarray(normals, dtype=jnp.float64)
    offsets64 = jnp.asarray(offsets, dtype=jnp.float64)
    try:
        vertices_np = _spatial.halfspace_intersection_vertices(normals64, offsets64)
        vertices = jnp.asarray(vertices_np, dtype=jnp.float64)
        if vertices.shape[0] == 0:
            raise ValueError("no vertices")
    except (ValueError, RuntimeError):
        dimension = int(normals64.shape[1]) if normals64.ndim == 2 else 0
        vertices = jnp.empty((0, dimension), dtype=jnp.float64)
    return vertices


def hypercube(
    dimension: int, *, radius: float = 1.0
) -> tuple[
    Float[Array, " num_vertices dimension"],
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]:
    """Axis-aligned hypercube centred at the origin; returns (verts, normals, offsets)."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    corners = jnp.array(
        list(itertools.product((-radius, radius), repeat=dimension)), dtype=jnp.float64
    )
    return from_vertices(corners)


def cross_polytope(
    dimension: int, *, radius: float = 1.0
) -> tuple[
    Float[Array, " num_vertices dimension"],
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]:
    """L1-ball (cross polytope); returns (verts, normals, offsets)."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    basis = jnp.eye(dimension, dtype=jnp.float64)
    vertices = jnp.concatenate([radius * basis, -radius * basis], axis=0)
    return from_vertices(vertices)


def simplex(
    dimension: int,
) -> tuple[
    Float[Array, " num_vertices dimension"],
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]:
    """Standard simplex in ``dimension`` dimensions; returns (verts, normals, offsets)."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    basis = jnp.eye(dimension, dtype=jnp.float64)
    origin = jnp.zeros((1, dimension), dtype=jnp.float64)
    vertices = jnp.concatenate([basis, origin], axis=0)
    return from_vertices(vertices)


def sample_halfspace(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_facets: int,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[
    tuple[
        Float[Array, " num_vertices dimension"],
        Float[Array, " num_facets dimension"],
        Float[Array, " num_facets"],
    ],
    ...,
]:
    """Sample bounded polytopes from random half-space inequalities."""

    samples: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    attempts = 0
    rng = key
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, normals_key = jax.random.split(rng)
        normals_key, offsets_key = jax.random.split(normals_key)
        normals = jax.random.normal(normals_key, (num_facets, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.where(norms == 0.0, 1.0, norms)
        offsets = jax.random.uniform(
            offsets_key,
            (num_facets,),
            minval=0.5,
            maxval=2.0,
            dtype=jnp.float64,
        )
        vertices = from_halfspaces(normals, offsets)
        if vertices.shape[0] == 0:
            continue
        samples.append((vertices, normals, offsets))
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough bounded polytopes from half-space data")
    return tuple(samples)


def sample_halfspace_tangent(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_facets: int,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[
    tuple[
        Float[Array, " num_vertices dimension"],
        Float[Array, " num_facets dimension"],
        Float[Array, " num_facets"],
    ],
    ...,
]:
    """Sample polytopes from half-spaces tangent to the unit ball."""

    samples: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    attempts = 0
    rng = key
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, normals_key = jax.random.split(rng)
        normals = jax.random.normal(normals_key, (num_facets, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.where(norms == 0.0, 1.0, norms)
        offsets = jnp.ones((num_facets,), dtype=jnp.float64)
        vertices = from_halfspaces(normals, offsets)
        if vertices.shape[0] == 0:
            continue
        if not _contains_origin_offsets(offsets):
            continue
        samples.append((vertices, normals, offsets))
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough tangent polytopes")
    return tuple(samples)


def sample_uniform_sphere(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[
    tuple[
        Float[Array, " num_vertices dimension"],
        Float[Array, " num_facets dimension"],
        Float[Array, " num_facets"],
    ],
    ...,
]:
    """Sample convex hulls of points drawn uniformly from the unit sphere."""

    samples: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    attempts = 0
    rng = key
    num_vertices = dimension + 1
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, draw_key = jax.random.split(rng)
        points = jax.random.normal(draw_key, (num_vertices, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(points, axis=1, keepdims=True)
        vertices = points / jnp.where(norms == 0.0, 1.0, norms)
        hull_vertices, normals, offsets = from_vertices(vertices)
        if not _contains_origin_offsets(offsets):
            continue
        samples.append((hull_vertices, normals, offsets))
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough sphere polytopes that contain the origin")
    return tuple(samples)


def sample_uniform_ball(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[
    tuple[
        Float[Array, " num_vertices dimension"],
        Float[Array, " num_facets dimension"],
        Float[Array, " num_facets"],
    ],
    ...,
]:
    """Sample convex hulls of points drawn uniformly from the unit ball."""

    samples: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    attempts = 0
    rng = key
    num_vertices = dimension + 1
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, draw_key = jax.random.split(rng)
        draw_key, radius_key = jax.random.split(draw_key)
        directions = jax.random.normal(draw_key, (num_vertices, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
        unit_dirs = directions / jnp.where(norms == 0.0, 1.0, norms)
        radii = jax.random.uniform(radius_key, (num_vertices, 1), dtype=jnp.float64)
        vertices = unit_dirs * jnp.power(radii, 1.0 / dimension)
        hull_vertices, normals, offsets = from_vertices(vertices)
        if not _contains_origin_offsets(offsets):
            continue
        samples.append((hull_vertices, normals, offsets))
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough ball polytopes that contain the origin")
    return tuple(samples)


def enumerate_product_ngons(
    max_ngon_P: int,
    max_ngon_Q: int,
    max_rotation_Q: int,
) -> tuple[
    tuple[
        Float[Array, " num_vertices 4"],
        Float[Array, " num_facets 4"],
        Float[Array, " num_facets"],
    ],
    ...,
]:
    """Enumerate Cartesian products of regular polygons with rotation grid."""

    samples: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    for k_P in range(3, max_ngon_P + 1):
        for k_Q in range(3, max_ngon_Q + 1):
            for s in range(1, max_rotation_Q + 1):
                for r in range(0, s):
                    if r == 0 and s != 1:
                        continue
                    if r != 0 and math.gcd(r, s) != 1:
                        continue
                    if r / s >= 1.0 / k_Q:
                        continue
                    angle = 2.0 * math.pi * r / s
                    # Vertex sets for P and Q not required for half-space assembly.
                    # Vertices of the Cartesian product are not needed here; we use
                    # the half-space description to derive the hull vertices.
                    normals_P = jnp.array(
                        [
                            [
                                math.cos((2.0 * math.pi * i / k_P) + math.pi / k_P),
                                math.sin((2.0 * math.pi * i / k_P) + math.pi / k_P),
                            ]
                            for i in range(k_P)
                        ],
                        dtype=jnp.float64,
                    )
                    normals_Q = jnp.array(
                        [
                            [
                                math.cos((2.0 * math.pi * i / k_Q) + math.pi / k_Q + angle),
                                math.sin((2.0 * math.pi * i / k_Q) + math.pi / k_Q + angle),
                            ]
                            for i in range(k_Q)
                        ],
                        dtype=jnp.float64,
                    )
                    offsets_P = jnp.full((k_P,), math.cos(math.pi / k_P), dtype=jnp.float64)
                    offsets_Q = jnp.full((k_Q,), math.cos(math.pi / k_Q), dtype=jnp.float64)
                    normals = jnp.concatenate(
                        (
                            jnp.concatenate(
                                (normals_P, jnp.zeros((k_P, 2), dtype=jnp.float64)),
                                axis=1,
                            ),
                            jnp.concatenate(
                                (jnp.zeros((k_Q, 2), dtype=jnp.float64), normals_Q),
                                axis=1,
                            ),
                        ),
                        axis=0,
                    )
                    offsets = jnp.concatenate((offsets_P, offsets_Q), axis=0)
                    hull_vertices = from_halfspaces(normals, offsets)
                    samples.append((hull_vertices, normals, offsets))
    return tuple(samples)


def _contains_origin_offsets(
    offsets: Float[Array, " num_facets"], *, atol: float = GEOMETRY_ABS_TOLERANCE
) -> bool:
    if offsets.size == 0:
        return False
    return bool(jnp.all(offsets >= -float(atol)))


def regular_ngon(
    k: int,
    *,
    radius: float = 1.0,
    rotation: float = 0.0,
) -> tuple[
    Float[Array, " num_vertices 2"],
    Float[Array, " num_facets 2"],
    Float[Array, " num_facets"],
]:
    """Return a regular k-gon in 2D as (vertices, normals, offsets).

    Vertices are placed on a circle of given ``radius`` with a global
    counter-clockwise rotation ``rotation`` (radians). The half-space
    description is derived from the convex hull of the vertices to remain
    robust and consistent with other helpers.
    """

    if k < 3:
        raise ValueError("regular_ngon requires k >= 3")
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num=k, endpoint=False, dtype=jnp.float64) + float(
        rotation
    )
    vertices = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1) * jnp.asarray(
        radius, dtype=jnp.float64
    )
    return from_vertices(vertices)


def rotate_2d_vertices(vertices: Float[Array, " n 2"], *, theta: float) -> Float[Array, " n 2"]:
    """Rotate 2D vertices by angle ``theta`` (radians), counter-clockwise."""

    R = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]], dtype=jnp.float64
    )
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    return verts @ R.T


def rotate_2d_halfspaces(
    B: Float[Array, " m 2"],
    c: Float[Array, " m"],
    *,
    theta: float,
) -> tuple[Float[Array, " m 2"], Float[Array, " m"]]:
    """Rotate 2D half-spaces ``Bx ≤ c`` by angle ``theta``.

    Returns the rotated matrix ``B'`` and unchanged offsets ``c`` such that
    ``{x | Bx ≤ c}`` rotated by ``R`` equals ``{x | B' x ≤ c}`` with
    ``B' = B Rᵀ`` (for orthogonal rotation ``R``).
    """

    R = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]], dtype=jnp.float64
    )
    Bm = jnp.asarray(B, dtype=jnp.float64)
    cm = jnp.asarray(c, dtype=jnp.float64)
    return Bm @ R.T, cm


def product_halfspaces(
    B_left: Float[Array, " m1 d1"],
    c_left: Float[Array, " m1"],
    B_right: Float[Array, " m2 d2"],
    c_right: Float[Array, " m2"],
) -> tuple[
    Float[Array, " m  d"],
    Float[Array, " m"],
]:
    """Cartesian product of two polytopes given as half-spaces.

    For ``K = {x | B_l x ≤ c_l} ⊂ R^{d1}`` and ``L = {y | B_r y ≤ c_r} ⊂ R^{d2}``,
    returns a half-space description of ``K × L ⊂ R^{d1+d2}``.
    """

    Bl = jnp.asarray(B_left, dtype=jnp.float64)
    cl = jnp.asarray(c_left, dtype=jnp.float64)
    Br = jnp.asarray(B_right, dtype=jnp.float64)
    cr = jnp.asarray(c_right, dtype=jnp.float64)
    d1 = int(Bl.shape[1])
    d2 = int(Br.shape[1])
    left_block = jnp.concatenate((Bl, jnp.zeros((Bl.shape[0], d2), dtype=jnp.float64)), axis=1)
    right_block = jnp.concatenate((jnp.zeros((Br.shape[0], d1), dtype=jnp.float64), Br), axis=1)
    B_prod = jnp.concatenate((left_block, right_block), axis=0)
    c_prod = jnp.concatenate((cl, cr), axis=0)
    return B_prod, c_prod


def pentagon_product_4d(
    *,
    rotation: float = math.pi / 2.0,
    area: float | None = None,
    radius: float | None = None,
) -> tuple[
    Float[Array, " num_vertices 4"],
    Float[Array, " num_facets 4"],
    Float[Array, " num_facets"],
]:
    """Return the 4D Cartesian product of two congruent regular pentagons.

    By default, the second pentagon is rotated by ``rotation`` (radians) in its
    plane. Use either ``area`` to specify each pentagon's area, or ``radius`` to
    specify the circumradius; if both are ``None``, uses ``radius=1.0``.

    This family provides a convenient constructor for the 4D pentagon products
    featured in recent literature on the Viterbo counterexample (Haim‑Kislev &
    Ostrover, 2024). Rotation and normalisation remain caller‑controlled.
    """

    if radius is None:
        if area is None:
            radius = 1.0
        else:
            # Area of a regular k-gon with circumradius r: (k/2) r^2 sin(2π/k)
            k = 5
            radius = float(jnp.sqrt(2.0 * area / (k * jnp.sin(2.0 * jnp.pi / k))))

    # Left pentagon (no rotation)
    _, B_L, c_L = regular_ngon(5, radius=float(radius), rotation=0.0)
    # Right pentagon (rotated)
    _, B_R_raw, c_R = regular_ngon(5, radius=float(radius), rotation=0.0)
    B_R, c_R = rotate_2d_halfspaces(B_R_raw, c_R, theta=float(rotation))

    # Cartesian product in R^4
    B4, c4 = product_halfspaces(B_L, c_L, B_R, c_R)
    verts4 = from_halfspaces(B4, c4)
    return verts4, B4, c4


def counterexample_hk_ostrover_4d(
    *, rotation: float = math.pi / 2.0, area: float | None = None
) -> tuple[
    Float[Array, " num_vertices 4"],
    Float[Array, " num_facets 4"],
    Float[Array, " num_facets"],
]:
    """Convenience wrapper for a congruent-pentagon product in 4D.

    Uses a small rotation by default (π/10); callers may specify a preferred
    normalisation via ``area``. Returns (vertices, normals, offsets).
    """

    return pentagon_product_4d(rotation=rotation, area=area)
