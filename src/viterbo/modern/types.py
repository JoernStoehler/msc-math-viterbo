"""Shared dataclasses describing modern polytope artefacts.

The classes in this module are lightweight containers that will become pytrees
once we wire them into JAX transformations. They intentionally avoid providing
behaviour today; instead, they document how we expect to pass information
between pure math routines and dataset adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Array, Float, Bool


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


@dataclass(slots=True)
class Polytope:
    """Aggregated geometric description of a polytope.
    
    Attributes:
        normals: Outward normals of the facets.
        offsets: Right-hand side offsets aligned with ``normals``.
        vertices: Vertices of the polytope.
        incidence: Vertex-facet incidence matrix.
    """
    
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
    """A representative periodic Reeb orbit (cycle) on the polytope boundary.

    Attributes:
        points: Points that define the piecewise-linear cycle.
        incidence: Incidence of points to facets.
    """

    points: Float[Array, " num_points dimension"]
    incidence: Bool[Array, " num_points num_facets"]
