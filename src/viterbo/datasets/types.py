"""Shared dataclasses describing modern polytope artefacts (datasets layer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

from jaxtyping import Array, Bool, Float

HalfspaceData: TypeAlias = tuple[
    Float[Array, " num_facets dimension"],
    Float[Array, " num_facets"],
]
"""Tuple alias encoding ``(normals, offsets)`` for half-space geometry."""


VertexCloud: TypeAlias = Float[Array, " num_vertices dimension"]
"""Alias describing an unordered vertex cloud of a convex polytope."""


IncidenceMatrix: TypeAlias = Bool[Array, " num_vertices num_facets"]
"""Alias for the vertex–facet incidence matrix."""


MilpCapacityBounds: TypeAlias = tuple[float, float, int, str]
"""Tuple encoding ``(lower_bound, upper_bound, iterations, status)``."""


SupportRelaxationSummary: TypeAlias = tuple[
    float,
    Float[Array, " num_samples dimension"],
    Float[Array, " num_samples"],
    int,
]
"""Tuple encoding ``(capacity_upper_bound, directions, support_values, iterations)``."""


@dataclass(slots=True)
class Polytope:
    """Geometry-only polytope container.

    The dataclass stores the canonical ``(B, c, V, I)`` arrays without bundling
    provenance or solver-specific artefacts. Algorithms should prefer the raw
    tuple aliases above to keep signatures explicit, constructing ``Polytope``
    instances only when row-oriented records are necessary (e.g. datasets).
    """

    normals: Float[Array, " num_facets dimension"]
    offsets: Float[Array, " num_facets"]
    vertices: VertexCloud
    incidence: IncidenceMatrix

    @property
    def dimension(self) -> int:
        """Ambient dimension d for the polytope."""
        return int(self.normals.shape[1]) if self.normals.ndim == 2 else 0

    @property
    def facets(self) -> int:
        """Number of facets m in the half-space description."""
        return int(self.normals.shape[0]) if self.normals.ndim == 2 else 0

    def halfspace_data(self) -> HalfspaceData:
        """Return ``(normals, offsets)`` as detached arrays."""

        return self.normals.copy(), self.offsets.copy()

    def with_vertices(self, vertices: VertexCloud, incidence: IncidenceMatrix) -> "Polytope":
        """Return a new geometry bundle with updated vertex data."""

        return Polytope(
            normals=self.normals,
            offsets=self.offsets,
            vertices=vertices,
            incidence=incidence,
        )


@dataclass(slots=True)
class PolytopeMetadata:
    """Auxiliary metadata describing provenance for a polytope geometry."""

    slug: str
    description: str = ""
    reference_capacity: float | None = None


class PolytopeRecord(NamedTuple):
    """Pair bundling geometry with metadata without mixing the concerns."""

    geometry: Polytope
    metadata: PolytopeMetadata


class FacetPairing(NamedTuple):
    """Symmetry pairing summary for opposite polytope facets."""

    pairs: tuple[tuple[int, int], ...]
    unpaired: tuple[int, ...]


@dataclass(slots=True)
class NormalCone:
    """Normal cone data attached to a polytope vertex."""

    vertex: Float[Array, " dimension"]
    active_facets: tuple[int, ...]
    normals: Float[Array, " num_active dimension"]


@dataclass(slots=True)
class PolytopeCombinatorics:
    """Cached combinatorial structure derived from a polytope."""

    vertices: VertexCloud
    facet_adjacency: Bool[Array, " num_facets num_facets"]
    normal_cones: tuple[NormalCone, ...]


@dataclass(slots=True)
class Cycle:
    """A representative periodic Reeb orbit (cycle) on the polytope boundary."""

    points: Float[Array, " num_points dimension"]
    incidence: Bool[Array, " num_points num_facets"]


__all__ = [
    "HalfspaceData",
    "VertexCloud",
    "IncidenceMatrix",
    "MilpCapacityBounds",
    "SupportRelaxationSummary",
    "Polytope",
    "PolytopeMetadata",
    "PolytopeRecord",
    "FacetPairing",
    "NormalCone",
    "PolytopeCombinatorics",
    "Cycle",
]
