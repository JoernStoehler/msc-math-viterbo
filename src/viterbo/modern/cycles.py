"""Cycle extraction stubs for the modern API."""

from __future__ import annotations

from jaxtyping import Array, Float

from .types import PolytopeBundle


def minimum_cycle_reference(bundle: PolytopeBundle) -> Float[Array, " num_points dimension"]:
    """Return a representative minimum-action cycle."""

    raise NotImplementedError


def minimum_cycle_batched(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    padded_vertices: Float[Array, " batch num_vertices dimension"],
) -> Float[Array, " batch num_points dimension"]:
    """Return padded cycles for each polytope in the batch."""

    raise NotImplementedError
