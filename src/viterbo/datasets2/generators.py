"""Convenience polytope generators emitting a unified data container."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from viterbo.math import generators as _gen
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


def _from_vertices(  # pyright: ignore[reportUnusedFunction]
    vertices: Float[Array, " num_vertices dimension"],
) -> PolytopeSample:
    hull_vertices, normals, offsets = _gen.from_vertices(vertices)
    incidence = _incidence(normals, offsets, hull_vertices)
    return PolytopeSample(hull_vertices, normals, offsets, incidence)


def _from_halfspaces(  # pyright: ignore[reportUnusedFunction]
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> PolytopeSample:
    vertices = _gen.from_halfspaces(normals, offsets, atol=atol)
    normals64 = jnp.asarray(normals, dtype=jnp.float64)
    offsets64 = jnp.asarray(offsets, dtype=jnp.float64)
    incidence = _incidence(normals64, offsets64, vertices)
    return PolytopeSample(vertices, normals64, offsets64, incidence)


def _contains_origin(  # pyright: ignore[reportUnusedFunction]
    sample: PolytopeSample, *, atol: float = GEOMETRY_ABS_TOLERANCE
) -> bool:
    offsets = sample.offsets
    if offsets.size == 0:
        return False
    return bool(jnp.all(offsets >= -float(atol)))


def hypercube(dimension: int, *, radius: float = 1.0) -> PolytopeSample:
    """Axis-aligned hypercube centred at the origin."""
    verts, normals, offsets = _gen.hypercube(dimension, radius=radius)
    incidence = _incidence(normals, offsets, verts)
    return PolytopeSample(verts, normals, offsets, incidence)


def cross_polytope(dimension: int, *, radius: float = 1.0) -> PolytopeSample:
    """L1-ball (cross polytope) in ``dimension`` dimensions."""
    verts, normals, offsets = _gen.cross_polytope(dimension, radius=radius)
    incidence = _incidence(normals, offsets, verts)
    return PolytopeSample(verts, normals, offsets, incidence)


def simplex(dimension: int) -> PolytopeSample:
    """Standard simplex in ``dimension`` dimensions."""
    verts, normals, offsets = _gen.simplex(dimension)
    incidence = _incidence(normals, offsets, verts)
    return PolytopeSample(verts, normals, offsets, incidence)


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
    for vertices, normals, offsets in _gen.sample_halfspace(
        key, dimension, num_facets=num_facets, num_samples=num_samples, max_attempts=max_attempts
    ):
        incidence = _incidence(normals, offsets, vertices)
        samples.append(PolytopeSample(vertices, normals, offsets, incidence))
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
    for vertices, normals, offsets in _gen.sample_halfspace_tangent(
        key, dimension, num_facets=num_facets, num_samples=num_samples, max_attempts=max_attempts
    ):
        incidence = _incidence(normals, offsets, vertices)
        samples.append(PolytopeSample(vertices, normals, offsets, incidence))
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
    for vertices, normals, offsets in _gen.sample_uniform_sphere(
        key, dimension, num_samples=num_samples, max_attempts=max_attempts
    ):
        incidence = _incidence(normals, offsets, vertices)
        samples.append(PolytopeSample(vertices, normals, offsets, incidence))
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
    for vertices, normals, offsets in _gen.sample_uniform_ball(
        key, dimension, num_samples=num_samples, max_attempts=max_attempts
    ):
        incidence = _incidence(normals, offsets, vertices)
        samples.append(PolytopeSample(vertices, normals, offsets, incidence))
    return tuple(samples)


def enumerate_product_ngons(
    max_ngon_P: int,
    max_ngon_Q: int,
    max_rotation_Q: int,
) -> tuple[PolytopeSample, ...]:
    """Enumerate Cartesian products of regular polygons with rotation grid."""

    samples: list[PolytopeSample] = []
    for vertices, normals, offsets in _gen.enumerate_product_ngons(
        max_ngon_P, max_ngon_Q, max_rotation_Q
    ):
        incidence = _incidence(normals, offsets, vertices)
        samples.append(PolytopeSample(vertices, normals, offsets, incidence))
    return tuple(samples)
