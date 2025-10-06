"""Direction sampling and smoothing kernels for support relaxations."""

# The module now offers both convex-combination and softmax-based smoothing
# kernels so solvers can choose between conservative but cheap averaging and a
# temperature-controlled variant that adapts to anisotropic support functions
# while retaining monotone upper bounds.

from __future__ import annotations

from typing import Iterable, Literal

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from viterbo.symplectic.core import standard_symplectic_matrix

SmoothingMethod = Literal["convex", "softmax"]


def validate_vertices(
    vertices: Float[Array, " num_vertices dimension"],
) -> Float[Array, " num_vertices dimension"]:
    """Validate vertex array shape and ensure an even ambient dimension."""
    vertices = jnp.asarray(vertices, dtype=jnp.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        msg = "Support relaxation expects a non-empty (num_vertices, dimension) array."
        raise ValueError(msg)
    if vertices.shape[1] % 2 != 0:
        msg = "Support relaxation requires an even ambient dimension."
        raise ValueError(msg)
    return vertices


def grid_directions(
    dimension: int,
    *,
    density: int,
) -> Float[Array, " num_directions dimension"]:
    """Return approximately uniform directions on the d-dimensional cube grid."""
    if dimension <= 0:
        msg = "Dimension must be positive."
        raise ValueError(msg)
    if density < 2:
        msg = "Grid density must be at least 2."
        raise ValueError(msg)

    axes = [
        np.linspace(-1.0, 1.0, density, dtype=np.float64) for _ in range(dimension)
    ]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
    norms = np.linalg.norm(mesh, axis=1)
    mask = norms > 1e-12
    filtered = mesh[mask]
    norms = norms[mask][:, None]
    normalised = filtered / norms
    # Remove duplicate rows caused by the projection of antipodal points.
    normalised = np.unique(np.round(normalised, decimals=12), axis=0)
    result: Float[Array, " num_directions dimension"] = jnp.asarray(
        normalised, dtype=jnp.float64
    )
    return result


def support_products(
    vertices: Float[Array, " num_vertices dimension"],
    directions: Float[Array, " num_directions dimension"],
) -> Float[Array, " num_directions"]:
    """Return ``h_K(u) * h_K(Ju)`` for all sampled directions."""
    vertices = validate_vertices(vertices)
    directions = jnp.asarray(directions, dtype=jnp.float64)
    if directions.ndim != 2 or directions.shape[0] == 0:
        msg = "Direction grid must be two-dimensional and non-empty."
        raise ValueError(msg)
    if directions.shape[1] != vertices.shape[1]:
        msg = "Directions and vertices must share the same ambient dimension."
        raise ValueError(msg)

    symplectic = standard_symplectic_matrix(vertices.shape[1])

    support_values = jnp.max(vertices @ directions.T, axis=0)
    rotated = directions @ symplectic.T
    rotated_support = jnp.max(vertices @ rotated.T, axis=0)
    products: Float[Array, " num_directions"] = support_values * rotated_support
    return products


def smoothing_strength(parameter: float) -> float:
    """Map a smoothing parameter to a convex-combination weight in ``[0, 1)``."""
    if parameter < 0.0:
        msg = "Smoothing parameter must be non-negative."
        raise ValueError(msg)
    strength = 1.0 - float(np.exp(-parameter))
    return strength


def smooth_support_products(
    products: Float[Array, " num_directions"],
    *,
    strength: float,
) -> Float[Array, " num_directions"]:
    """Interpolate support products towards their global maximum."""
    return smooth_support_products_with_method(
        products, strength=strength, method="convex"
    )


def smooth_support_products_with_method(
    products: Float[Array, " num_directions"],
    *,
    strength: float,
    method: SmoothingMethod,
) -> Float[Array, " num_directions"]:
    """Interpolate support products using the requested smoothing kernel."""
    if not 0.0 <= strength < 1.0:
        msg = "Smoothing strength must lie in [0, 1)."
        raise ValueError(msg)
    products = jnp.asarray(products, dtype=jnp.float64)
    maximum = jnp.max(products)
    if method == "convex":
        smoothed = (1.0 - strength) * products + strength * maximum
        return smoothed
    if method == "softmax":
        temperature = float(max(1e-12, 1.0 - strength))
        shifted = (products - maximum) / temperature
        weights = jnn.softmax(shifted)
        weighted_average = jnp.sum(weights * products)
        mixture = (1.0 - strength) * products + strength * weighted_average
        smoothed = jnp.maximum(products, mixture)
        smoothed = jnp.minimum(smoothed, maximum)
        return smoothed
    msg = f"Unknown smoothing method '{method}'."
    raise ValueError(msg)


def continuation_schedule(
    parameters: Iterable[float],
) -> tuple[float, ...]:
    """Return a monotone non-increasing schedule suitable for continuation."""
    values = tuple(float(p) for p in parameters)
    if any(p < 0.0 for p in values):
        msg = "Continuation parameters must be non-negative."
        raise ValueError(msg)
    return tuple(sorted(values, reverse=True))
