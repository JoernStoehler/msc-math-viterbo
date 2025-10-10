"""EHZ capacity routines for the modern API."""

from __future__ import annotations

from jaxtyping import Array, Float

from .types import PolytopeBundle


def ehz_capacity_reference(bundle: PolytopeBundle) -> float:
    """Return a reference EHZ capacity estimate."""

    raise NotImplementedError


def ehz_capacity_batched(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    max_cycles: int,
) -> Float[Array, " batch"]:
    """Compute batched EHZ capacities with padding."""

    raise NotImplementedError
