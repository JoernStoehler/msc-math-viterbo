"""Shared dataclasses describing modern polytope artefacts.

The classes in this module are lightweight containers that will become pytrees
once we wire them into JAX transformations. They intentionally avoid providing
behaviour today; instead, they document how we expect to pass information
between pure math routines and dataset adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, Float


@dataclass(slots=True)
class HalfspaceGeometry:
    """Half-space representation ``Bx ≤ c`` of a convex polytope.

    Attributes:
      normals: Outward normals of the facets.
      offsets: Right-hand side offsets aligned with ``normals``.
    """

    normals: Float[Array, " num_facets dimension"]
    offsets: Float[Array, " num_facets"]


@dataclass(slots=True)
class VertexGeometry:
    """Vertex representation of a convex polytope."""

    vertices: Float[Array, " num_vertices dimension"]
    incidence: Float[Array, " num_vertices num_facets"] | None = None


@dataclass(slots=True)
class PolytopeBundle:
    """Aggregated geometric description of a polytope.

    Attributes:
      halfspaces: Optional half-space data, when available.
      vertices: Optional vertex data, when available.
    """

    halfspaces: HalfspaceGeometry | None
    vertices: VertexGeometry | None


@dataclass(slots=True)
class GeneratorMetadata:
    """Description of the generator responsible for a polytope sample."""

    identifier: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class QuantityRecord:
    """Collection of computed quantities for a single polytope."""

    volume: float | None = None
    capacity_ehz: float | None = None
    spectrum_ehz: tuple[float, ...] | None = None
    cycle_path: Float[Array, " num_points dimension"] | None = None
