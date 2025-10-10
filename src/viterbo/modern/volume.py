"""Volume estimators for the modern API."""

from __future__ import annotations

from jaxtyping import Array, Float

from .types import PolytopeBundle


def volume_reference(bundle: PolytopeBundle) -> float:
    """Return a reference volume estimate for ``bundle``."""

    raise NotImplementedError


def volume_padded(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    method: str,
) -> Float[Array, " batch"]:
    """Compute batched volumes using a padding-friendly method."""

    raise NotImplementedError
