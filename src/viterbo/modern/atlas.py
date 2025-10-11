"""The global atlas dataset, with conversion helpers."""

from __future__ import annotations
from dataclasses import dataclass

import polars as pl
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.modern.types import Polytope, Cycle
from viterbo.modern.polytopes import incidence_matrix

# we use different schemas for different dimensions
# to allow fixed-size arrays, batching, etc.
def atlas_pl_schema(dimension: int) -> pl.Schema:
    """Schema for atlas rows at a fixed `dimension`.

    The schema encodes vector-valued columns as fixed-length Polars `Array`
    types to enable predictable padding and batching downstream.
    """
    VecF64 = pl.Array(pl.Float64, dimension)
    return pl.Schema(
        {
            # basic metadata
            "polytope_id": pl.String(),
            "notes": pl.String(),
            "distribution_name": pl.String(),
            # polytope representation
            "dimension": pl.Int64(),
            "num_facets": pl.Int64(),
            "num_vertices": pl.Int64(),
            "normals": pl.List(VecF64),
            "offsets": pl.List(pl.Float64),
            "vertices": pl.List(VecF64),
            # quantities
            "volume": pl.Float64(),
            "ehz_capacity": pl.Float64(),
            "systolic_ratio": pl.Float64(),
            "minimum_action_cycle": pl.List(VecF64),
        }
    )

@dataclass(slots=True)
class AtlasRow:
    """A single row of the atlas dataset, parsed into JAX types."""
    # polytope metadata
    polytope_id: str
    dimension: int
    notes: str
    distribution_name: str
    # polytope representation
    polytope: Polytope
    # quantities
    volume: Float[Array, ""]
    ehz_capacity: Float[Array, ""]
    systolic_ratio: Float[Array, ""]
    minimum_action_cycle: Cycle

def as_polytope(
    dimension: int,
    num_facets: int,
    num_vertices: int,
    normals: list[Float[Array, " dimension"]],
    offsets: list[float],
    vertices: list[Float[Array, " dimension"]],
) -> Polytope:
    """Convert row fields into a JAX-first `Polytope`.

    Ensures float64 dtype and validates shapes before computing the facet–
    vertex incidence via the half-space test.
    """
    _normals = jnp.asarray(normals, dtype=jnp.float64)
    _offsets = jnp.asarray(offsets, dtype=jnp.float64)
    _vertices = jnp.asarray(vertices, dtype=jnp.float64)
    assert _normals.shape == (num_facets, dimension)
    assert _offsets.shape == (num_facets,)
    assert _vertices.shape == (num_vertices, dimension)
    _incidence = incidence_matrix(_normals, _offsets, _vertices)
    return Polytope(
        normals=_normals,
        offsets=_offsets,
        vertices=_vertices,
        incidence=_incidence,
    )

def as_cycle(
    dimension: int,
    num_points: int,
    points: list[Float[Array, " dimension"]],
    polytope: Polytope,
) -> Cycle:
    """Convert row fields into a `Cycle` linked to `polytope`.

    Computes point–facet incidence with consistent tolerances used by the
    polytope helper.
    """
    _points = jnp.asarray(points, dtype=jnp.float64)
    assert _points.shape == (num_points, dimension)
    normals = polytope.normals
    offsets = polytope.offsets
    incidence = incidence_matrix(normals, offsets, _points)
    return Cycle(points=_points, incidence=incidence)
