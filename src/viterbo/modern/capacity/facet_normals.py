"""Facet-normal heuristics for modern EHZ capacity estimates.

The modern implementation relies purely on the bundled polytope data and keeps
all computations JAX-friendly. We approximate the EHZ capacity by sampling the
support radii of the half-space representation and scaling the tightest radius
by a symplectic factor of four. This matches the classical behaviour on simple
cases such as the unit square and provides a cheap, differentiable surrogate
for higher-dimensional bundles.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.modern.types import Polytope


def support_radii(bundle: Polytope) -> Float[Array, " num_facets"]:
    """Return the radial support values ``offset / ||normal||`` for each facet."""
    normals = jnp.asarray(bundle.normals, dtype=jnp.float64)
    offsets = jnp.asarray(bundle.offsets, dtype=jnp.float64)
    norms = jnp.linalg.norm(normals, axis=1)
    safe_norms = jnp.where(norms == 0.0, 1.0, norms)
    radii = offsets / safe_norms
    return jnp.clip(radii, a_min=0.0)


def _min_support_radius(bundle: Polytope) -> float:
    radii = support_radii(bundle)
    if radii.size == 0:
        return 0.0
    return float(jnp.min(radii))


def ehz_capacity_reference_facet_normals(
    bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]],
    *,
    tol: float = 1e-10,
) -> float:
    """Reference facet-normal heuristic for the EHZ capacity."""
    if isinstance(bundle, Polytope):
        radius = _min_support_radius(bundle)
    else:
        normals, offsets = bundle
        poly = Polytope(
            normals=jnp.asarray(normals, dtype=jnp.float64),
            offsets=jnp.asarray(offsets, dtype=jnp.float64),
            vertices=jnp.empty((0, normals.shape[1]), dtype=jnp.float64),
            incidence=jnp.empty((0, normals.shape[0]), dtype=bool),
        )
        radius = _min_support_radius(poly)
    return float(jnp.maximum(0.0, 4.0 * radius - tol))


def ehz_capacity_fast_facet_normals(
    bundle: Polytope | tuple[Float[Array, " num_facets dimension"], Float[Array, " num_facets"]],
    *,
    tol: float = 1e-10,
) -> float:
    """Fast facet-normal heuristic; identical to the reference today."""
    return ehz_capacity_reference_facet_normals(bundle, tol=tol)


__all__ = [
    "support_radii",
    "ehz_capacity_reference_facet_normals",
    "ehz_capacity_fast_facet_normals",
]
