"""Stub interfaces for constructing polytope geometry."""

from __future__ import annotations

from jaxtyping import Array, Float

from .types import PolytopeBundle


def build_from_halfspaces(
    normals: Float[Array, " num_facets dimension"],
    offsets: Float[Array, " num_facets"],
) -> PolytopeBundle:
    """Return a bundle populated from half-space data."""

    raise NotImplementedError


def build_from_vertices(
    vertices: Float[Array, " num_vertices dimension"],
) -> PolytopeBundle:
    """Return a bundle populated from vertex data."""

    raise NotImplementedError


def complete_incidence(bundle: PolytopeBundle) -> PolytopeBundle:
    """Augment missing incidence information in ``bundle``."""

    raise NotImplementedError


def pad_polytope_bundle(
    bundle: PolytopeBundle,
    *,
    target_facets: int,
    target_vertices: int,
) -> PolytopeBundle:
    """Pad the bundle to fixed sizes for batching."""

    raise NotImplementedError
