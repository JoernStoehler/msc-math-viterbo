"""Accelerated solver for support-function relaxations of ``c_EHZ``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.symplectic.capacity.support_relaxation import kernels


@dataclass(frozen=True)
class SupportRelaxationDiagnostics:
    """Single iteration diagnostics for continuation and refinement."""

    grid_density: int
    smoothing_parameter: float
    smoothing_strength: float
    tolerance: float
    candidate_capacity: float


@dataclass(frozen=True)
class SupportRelaxationResult:
    """Result of the support-function relaxation algorithm."""

    capacity_upper_bound: float
    history: tuple[SupportRelaxationDiagnostics, ...]


def _prepare_vertices(
    vertices: Float[Array, " num_vertices dimension"],
    *,
    center_vertices: bool,
) -> Float[Array, " num_vertices dimension"]:
    vertices = kernels.validate_vertices(vertices)
    if center_vertices:
        centre = jnp.mean(vertices, axis=0)
        vertices = vertices - centre
    return vertices


def _compute_candidate(
    vertices: Float[Array, " num_vertices dimension"],
    directions: Float[Array, " num_directions dimension"],
    *,
    strength: float,
    method: kernels.SmoothingMethod,
) -> Float[Array, ""]:
    products = kernels.support_products(vertices, directions)
    smoothed = kernels.smooth_support_products_with_method(
        products,
        strength=strength,
        method=method,
    )
    candidate = jnp.pi * jnp.max(smoothed)
    return candidate


def compute_support_relaxation_capacity_fast(
    vertices: Float[Array, " num_vertices dimension"],
    *,
    initial_density: int = 7,
    refinement_steps: int = 3,
    refinement_growth: int = 2,
    smoothing_parameters: Iterable[float] = (0.9, 0.6, 0.3, 0.0),
    smoothing_method: kernels.SmoothingMethod = "convex",
    tolerance_sequence: Sequence[float] = (1e-3, 1e-4, 1e-5),
    log_callback: Callable[[SupportRelaxationDiagnostics], None] | None = None,
    center_vertices: bool = True,
    jit_compile: bool = True,
) -> SupportRelaxationResult:
    """Compute an upper bound on ``c_EHZ`` using adaptive support relaxations."""
    # ``smoothing_method`` selects the interpolation kernel used when tracing the
    # support products. ``"convex"`` retains the original convex-combination
    # behaviour while ``"softmax"`` leverages a temperature-controlled softmax
    # blend that preserves monotonicity and can tighten estimates on anisotropic
    # inputs.
    vertices = _prepare_vertices(vertices, center_vertices=center_vertices)

    smoothing_schedule = kernels.continuation_schedule(smoothing_parameters)
    tolerance_sequence = tuple(float(t) for t in tolerance_sequence)
    if not tolerance_sequence:
        tolerance_sequence = (0.0,)

    history: list[SupportRelaxationDiagnostics] = []
    best_value = float("inf")

    method = smoothing_method
    candidate_function = _compute_candidate
    if jit_compile:
        candidate_function = jax.jit(
            _compute_candidate,
            static_argnames=("strength", "method"),
        )

    for stage, parameter in enumerate(smoothing_schedule):
        strength = kernels.smoothing_strength(parameter)
        density = initial_density
        tolerance = tolerance_sequence[min(stage, len(tolerance_sequence) - 1)]
        previous = float("inf")

        for _ in range(refinement_steps):
            directions = kernels.grid_directions(vertices.shape[1], density=density)
            candidate_value = candidate_function(
                vertices,
                directions,
                strength=strength,
                method=method,
            )
            candidate = float(candidate_value)
            best_value = min(best_value, candidate)
            history.append(
                SupportRelaxationDiagnostics(
                    grid_density=density,
                    smoothing_parameter=parameter,
                    smoothing_strength=strength,
                    tolerance=tolerance,
                    candidate_capacity=best_value,
                )
            )
            if log_callback is not None:
                log_callback(history[-1])

            if previous - candidate <= tolerance * max(1.0, candidate):
                break

            previous = candidate
            density += refinement_growth

    result = SupportRelaxationResult(
        capacity_upper_bound=best_value,
        history=tuple(history),
    )
    return result
