"""Convenience polytope generators emitting a unified data container."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from viterbo._wrapped import spatial as _spatial
from viterbo.math.numerics import (
    GEOMETRY_ABS_TOLERANCE,
    INCIDENCE_ABS_TOLERANCE,
    INCIDENCE_REL_TOLERANCE,
)


@dataclass(frozen=True)
class PolytopeSample:
    """Container returned by all generators.

    The structure keeps vertices-first ordering to align with callers that
    primarily operate on vertex data before deriving half-space or incidence
    information. ``as_tuple`` exists purely for compatibility with legacy call
    sites that still expect positional unpacking. The dataset2 API otherwise
    favours attribute access for readability.
    """

    vertices: Float[Array, " num_vertices dimension"]
    normals: Float[Array, " num_facets dimension"]
    offsets: Float[Array, " num_facets"]
    incidence: Bool[Array, " num_facets num_vertices"]

    def as_tuple(
        self,
    ) -> tuple[
        Float[Array, " num_vertices dimension"],
        Float[Array, " num_facets dimension"],
        Float[Array, " num_facets"],
        Bool[Array, " num_facets num_vertices"],
    ]:
        """Return a tuple view of the sample for positional consumers."""

        return (self.vertices, self.normals, self.offsets, self.incidence)
def _incidence(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    vertices: Float[Array, " num_vertices dimension"],
    *,
    rtol: float = INCIDENCE_REL_TOLERANCE,
    atol: float = INCIDENCE_ABS_TOLERANCE,
) -> Bool[Array, " num_facets num_vertices"]:
    """Return facet-vertex incidence matrix."""

    support = jnp.matmul(normals, vertices.T)
    return jnp.isclose(support, offsets[:, None], rtol=rtol, atol=atol)


def _from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
) -> PolytopeSample:
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    equations = _spatial.convex_hull_equations(verts)
    normals = jnp.asarray(equations[:, :-1], dtype=jnp.float64)
    offsets = jnp.asarray(-equations[:, -1], dtype=jnp.float64)
    hull_indices = jnp.asarray(_spatial.convex_hull_vertices(verts), dtype=jnp.int32)
    hull_vertices = verts[hull_indices]
    incidence = _incidence(normals, offsets, hull_vertices)
    return PolytopeSample(hull_vertices, normals, offsets, incidence)


def _from_halfspaces(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> PolytopeSample:
    normals64 = jnp.asarray(normals, dtype=jnp.float64)
    offsets64 = jnp.asarray(offsets, dtype=jnp.float64)
    try:
        vertices_np = _spatial.halfspace_intersection_vertices(normals64, offsets64)
        vertices = jnp.asarray(vertices_np, dtype=jnp.float64)
        if vertices.shape[0] == 0:
            raise ValueError("no vertices")
    except (ValueError, RuntimeError):
        dimension = int(normals64.shape[1])
        vertices = jnp.empty((0, dimension), dtype=jnp.float64)
    incidence = _incidence(normals64, offsets64, vertices)
    return PolytopeSample(vertices, normals64, offsets64, incidence)


def _contains_origin(sample: PolytopeSample, *, atol: float = GEOMETRY_ABS_TOLERANCE) -> bool:
    offsets = sample.offsets
    if offsets.size == 0:
        return False
    return bool(jnp.all(offsets >= -float(atol)))


def hypercube(dimension: int, *, radius: float = 1.0) -> PolytopeSample:
    """Axis-aligned hypercube centred at the origin."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    corners = jnp.array(
        list(itertools.product((-radius, radius), repeat=dimension)), dtype=jnp.float64
    )
    return _from_vertices(corners)


def cross_polytope(dimension: int, *, radius: float = 1.0) -> PolytopeSample:
    """L1-ball (cross polytope) in ``dimension`` dimensions."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    basis = jnp.eye(dimension, dtype=jnp.float64)
    vertices = jnp.concatenate([radius * basis, -radius * basis], axis=0)
    return _from_vertices(vertices)


def simplex(dimension: int) -> PolytopeSample:
    """Standard simplex in ``dimension`` dimensions."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    basis = jnp.eye(dimension, dtype=jnp.float64)
    origin = jnp.zeros((1, dimension), dtype=jnp.float64)
    vertices = jnp.concatenate([basis, origin], axis=0)
    return _from_vertices(vertices)


def sample_halfspace(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_facets: int,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[PolytopeSample, ...]:
    """Sample bounded polytopes from random half-space inequalities."""

    samples: list[PolytopeSample] = []
    attempts = 0
    rng = key
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, normals_key = jax.random.split(rng)
        normals_key, offsets_key = jax.random.split(normals_key)
        normals = jax.random.normal(
            normals_key, (num_facets, dimension), dtype=jnp.float64
        )
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.where(norms == 0.0, 1.0, norms)
        offsets = jax.random.uniform(
            offsets_key,
            (num_facets,),
            minval=0.5,
            maxval=2.0,
            dtype=jnp.float64,
        )
        sample = _from_halfspaces(normals, offsets)
        if sample.vertices.shape[0] == 0:
            continue
        samples.append(sample)
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
) -> tuple[PolytopeSample, ...]:
    """Sample polytopes from half-spaces tangent to the unit ball."""

    samples: list[PolytopeSample] = []
    attempts = 0
    rng = key
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, normals_key = jax.random.split(rng)
        normals = jax.random.normal(
            normals_key, (num_facets, dimension), dtype=jnp.float64
        )
        norms = jnp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / jnp.where(norms == 0.0, 1.0, norms)
        offsets = jnp.ones((num_facets,), dtype=jnp.float64)
        sample = _from_halfspaces(normals, offsets)
        if sample.vertices.shape[0] == 0:
            continue
        if not _contains_origin(sample):
            continue
        samples.append(sample)
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough tangent polytopes")
    return tuple(samples)


def sample_uniform_sphere(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[PolytopeSample, ...]:
    """Sample convex hulls of points drawn uniformly from the unit sphere."""

    samples: list[PolytopeSample] = []
    attempts = 0
    rng = key
    num_vertices = dimension + 1
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        rng, draw_key = jax.random.split(rng)
        points = jax.random.normal(draw_key, (num_vertices, dimension), dtype=jnp.float64)
        norms = jnp.linalg.norm(points, axis=1, keepdims=True)
        vertices = points / jnp.where(norms == 0.0, 1.0, norms)
        sample = _from_vertices(vertices)
        if not _contains_origin(sample):
            continue
        samples.append(sample)
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough sphere polytopes that contain the origin")
    return tuple(samples)


def sample_uniform_ball(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_samples: int,
    max_attempts: int = 64,
) -> tuple[PolytopeSample, ...]:
    """Sample convex hulls of points drawn uniformly from the unit ball."""

    samples: list[PolytopeSample] = []
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
        radii = jax.random.uniform(
            radius_key, (num_vertices, 1), dtype=jnp.float64
        )
        vertices = unit_dirs * jnp.power(radii, 1.0 / dimension)
        sample = _from_vertices(vertices)
        if not _contains_origin(sample):
            continue
        samples.append(sample)
    if len(samples) < num_samples:
        raise RuntimeError("Failed to sample enough ball polytopes that contain the origin")
    return tuple(samples)


def enumerate_product_ngons(
    max_ngon_P: int,
    max_ngon_Q: int,
    max_rotation_Q: int,
) -> tuple[PolytopeSample, ...]:
    """Enumerate Cartesian products of regular polygons with rotation grid."""

    samples: list[PolytopeSample] = []
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
                    vertices_P = jnp.array(
                        [
                            [math.cos(2.0 * math.pi * i / k_P), math.sin(2.0 * math.pi * i / k_P)]
                            for i in range(k_P)
                        ],
                        dtype=jnp.float64,
                    )
                    vertices_Q = jnp.array(
                        [
                            [
                                math.cos(2.0 * math.pi * i / k_Q + angle),
                                math.sin(2.0 * math.pi * i / k_Q + angle),
                            ]
                            for i in range(k_Q)
                        ],
                        dtype=jnp.float64,
                    )
                    vertices = jnp.array(
                        [
                            jnp.concatenate([v_P, v_Q])
                            for v_P in vertices_P
                            for v_Q in vertices_Q
                        ],
                        dtype=jnp.float64,
                    )
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
                    incidence = _incidence(normals, offsets, vertices)
                    samples.append(PolytopeSample(vertices, normals, offsets, incidence))
    return tuple(samples)


__all__ = [
    "PolytopeSample",
    "hypercube",
    "cross_polytope",
    "simplex",
    "sample_halfspace",
    "sample_halfspace_tangent",
    "sample_uniform_sphere",
    "sample_uniform_ball",
    "enumerate_product_ngons",
]
