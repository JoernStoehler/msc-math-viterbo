"""Reference solver relying on CVX for the support-function relaxation."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped import cvx
from viterbo.symplectic.capacity.support_relaxation import kernels
from viterbo.symplectic.capacity.support_relaxation.fast import (
    SupportRelaxationDiagnostics,
    SupportRelaxationResult,
)


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


def _solve_convex_epigraph(
    products: Float[Array, " num_directions"],
    *,
    solver: str,
    tolerance: float,
) -> float:
    objective = jnp.asarray(products, dtype=jnp.float64)
    value = cvx.solve_epigraph_minimum(objective, solver=solver, tolerance=tolerance)
    return float(value)


def compute_support_relaxation_capacity_reference(
    vertices: Float[Array, " num_vertices dimension"],
    *,
    grid_density: int = 9,
    smoothing_parameters: Iterable[float] = (1.0, 0.7, 0.4, 0.2, 0.0),
    tolerance_sequence: Sequence[float] = (1e-4, 1e-5, 1e-6),
    solver: str = "SCS",
    log_callback: Callable[[SupportRelaxationDiagnostics], None] | None = None,
    center_vertices: bool = True,
) -> SupportRelaxationResult:
    """Solve the relaxed convex program using CVXPy."""
    vertices = _prepare_vertices(vertices, center_vertices=center_vertices)

    smoothing_schedule = kernels.continuation_schedule(smoothing_parameters)
    tolerance_sequence = tuple(float(t) for t in tolerance_sequence)
    if not tolerance_sequence:
        tolerance_sequence = (0.0,)

    directions = kernels.grid_directions(vertices.shape[1], density=grid_density)
    base_products = kernels.support_products(vertices, directions)

    history: list[SupportRelaxationDiagnostics] = []
    best_value = float("inf")

    for stage, parameter in enumerate(smoothing_schedule):
        strength = kernels.smoothing_strength(parameter)
        tolerance = tolerance_sequence[min(stage, len(tolerance_sequence) - 1)]
        products = kernels.smooth_support_products(base_products, strength=strength)
        candidate = float(
            jnp.pi
            * _solve_convex_epigraph(
                products,
                solver=solver,
                tolerance=tolerance,
            )
        )
        best_value = min(best_value, candidate)
        diagnostics = SupportRelaxationDiagnostics(
            grid_density=grid_density,
            smoothing_parameter=parameter,
            smoothing_strength=strength,
            tolerance=tolerance,
            candidate_capacity=best_value,
        )
        history.append(diagnostics)
        if log_callback is not None:
            log_callback(diagnostics)

    result = SupportRelaxationResult(
        capacity_upper_bound=best_value,
        history=tuple(history),
    )
    return result
