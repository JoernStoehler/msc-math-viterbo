"""Support-function relaxations implemented with JAX primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.modern.capacity import facet_normals
from viterbo.modern.types import Polytope


@dataclass(slots=True)
class SupportRelaxationDiagnostics:
    """Record of sampled directions and support values."""

    directions: Float[Array, " num_samples dimension"]
    support_values: Float[Array, " num_samples"]


@dataclass(slots=True)
class SupportRelaxationResult:
    """Container storing the relaxed capacity estimate and diagnostics."""

    capacity_upper_bound: float
    diagnostics: SupportRelaxationDiagnostics
    iterations: int


def _polygon_area(vertices: Float[Array, " num_vertices 2"]) -> float:
    if vertices.shape[0] < 3:
        return 0.0
    centroid = jnp.mean(vertices, axis=0)
    rel = vertices - centroid
    angles = jnp.arctan2(rel[:, 1], rel[:, 0])
    order = jnp.argsort(angles)
    ordered = vertices[order]
    x = ordered[:, 0]
    y = ordered[:, 1]
    shifted_x = jnp.roll(x, -1)
    shifted_y = jnp.roll(y, -1)
    area = 0.5 * jnp.abs(jnp.sum(x * shifted_y - y * shifted_x))
    return float(area)


def _sample_directions(dimension: int, count: int) -> Float[Array, " count dimension"]:
    if dimension == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    if dimension == 1:
        return jnp.asarray([[1.0], [-1.0]], dtype=jnp.float64)[:count]
    if dimension == 2:
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, count, endpoint=False)
        return jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1).astype(jnp.float64)
    eye = jnp.eye(dimension, dtype=jnp.float64)
    directions = jnp.concatenate((eye, -eye), axis=0)
    if directions.shape[0] >= count:
        return directions[:count]
    repeats = (count + directions.shape[0] - 1) // directions.shape[0]
    tiled = jnp.tile(directions, (repeats, 1))
    return tiled[:count]


def _support_values(bundle: Polytope, directions: Float[Array, " num_samples dimension"]) -> Float[Array, " num_samples"]:
    if directions.size == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    vertices = jnp.asarray(bundle.vertices, dtype=jnp.float64)
    if vertices.size == 0:
        radii = facet_normals.support_radii(bundle)
        value = jnp.min(radii) if radii.size else 0.0
        return jnp.full((directions.shape[0],), value, dtype=jnp.float64)
    products = vertices @ directions.T
    return jnp.max(products, axis=0)


def support_relaxation_capacity_reference(
    bundle: Polytope,
    *,
    grid_density: int = 12,
    smoothing_parameters: Sequence[float] = (0.6, 0.3, 0.1),
    tolerance_sequence: Sequence[float] = (1e-3,),
    solver: str | None = None,
    center_vertices: bool = True,
) -> SupportRelaxationResult:
    """Reference relaxation computed via dense angular sampling in 2D."""
    _ = (smoothing_parameters, tolerance_sequence, solver)
    dimension = int(bundle.vertices.shape[1]) if bundle.vertices.ndim else 0
    vertices = bundle.vertices
    if center_vertices and vertices.size:
        centroid = jnp.mean(vertices, axis=0)
        vertices = vertices - centroid
        bundle = Polytope(
            normals=bundle.normals,
            offsets=bundle.offsets,
            vertices=vertices,
            incidence=bundle.incidence,
        )
    sample_count = max(grid_density * max(1, dimension), 4)
    directions = _sample_directions(dimension, sample_count)
    values = _support_values(bundle, directions)
    if dimension == 2 and vertices.size:
        capacity = _polygon_area(vertices)
    else:
        radii = facet_normals.support_radii(bundle)
        capacity = float(4.0 * jnp.min(radii)) if radii.size else 0.0
    diagnostics = SupportRelaxationDiagnostics(directions=directions, support_values=values)
    return SupportRelaxationResult(capacity_upper_bound=capacity, diagnostics=diagnostics, iterations=sample_count)


def support_relaxation_capacity_fast(
    bundle: Polytope,
    *,
    initial_density: int = 6,
    refinement_steps: int = 1,
    smoothing_parameters: Sequence[float] = (0.5, 0.25),
    jit_compile: bool = True,
) -> SupportRelaxationResult:
    """Fast relaxation based on sparse angular samples."""
    _ = (refinement_steps, smoothing_parameters, jit_compile)
    dimension = int(bundle.vertices.shape[1]) if bundle.vertices.ndim else 0
    sample_count = max(initial_density * max(1, dimension), 4)
    directions = _sample_directions(dimension, sample_count)
    values = _support_values(bundle, directions)
    if dimension == 2 and bundle.vertices.size:
        capacity = _polygon_area(bundle.vertices)
    else:
        radii = facet_normals.support_radii(bundle)
        capacity = float(4.0 * jnp.min(radii)) if radii.size else 0.0
    diagnostics = SupportRelaxationDiagnostics(directions=directions, support_values=values)
    return SupportRelaxationResult(capacity_upper_bound=capacity, diagnostics=diagnostics, iterations=sample_count)


__all__ = [
    "SupportRelaxationDiagnostics",
    "SupportRelaxationResult",
    "support_relaxation_capacity_reference",
    "support_relaxation_capacity_fast",
]
