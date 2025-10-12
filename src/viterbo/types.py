"""Shared dataclasses describing modern polytope artefacts (flat namespace)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, Float, Bool


@dataclass(slots=True)
class HalfspaceGeometry:
    """Half-space representation ``Bx ≤ c`` of a convex polytope."""

    normals: Float[Array, " num_facets dimension"]
    offsets: Float[Array, " num_facets"]


@dataclass(slots=True)
class VertexGeometry:
    """Vertex representation of a convex polytope."""

    vertices: Float[Array, " num_vertices dimension"]


@dataclass(slots=True)
class Polytope:
    """Aggregated geometric description of a polytope."""

    normals: Float[Array, " num_facets dimension"]
    offsets: Float[Array, " num_facets"]
    vertices: Float[Array, " num_vertices dimension"]
    incidence: Bool[Array, " num_vertices num_facets"]


@dataclass(slots=True)
class GeneratorMetadata:
    """Description of the generator responsible for a polytope sample."""

    identifier: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class Cycle:
    """A representative periodic Reeb orbit (cycle) on the polytope boundary."""

    points: Float[Array, " num_points dimension"]
    incidence: Bool[Array, " num_points num_facets"]


__all__ = [
    "HalfspaceGeometry",
    "VertexGeometry",
    "Polytope",
    "GeneratorMetadata",
    "Cycle",
]

