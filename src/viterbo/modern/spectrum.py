"""Discrete action spectrum routines for the modern API."""

from __future__ import annotations

from typing import Sequence

from jaxtyping import Array, Float

from .types import PolytopeBundle


def ehz_spectrum_reference(bundle: PolytopeBundle, *, head: int) -> Sequence[float]:
    """Return the leading entries of the EHZ action spectrum."""

    raise NotImplementedError


def ehz_spectrum_batched(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    head: int,
) -> Float[Array, " batch head"]:
    """Return padded EHZ spectra for each batch element."""

    raise NotImplementedError
