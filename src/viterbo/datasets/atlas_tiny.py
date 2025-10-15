"""AtlasTiny dataset helpers returning completed ragged rows.

This module assembles a deterministic roster of low-dimensional polytopes and
uses :mod:`viterbo.math` utilities to populate symplectic invariants. The
helpers return Python lists of typed dictionaries so callers can decide how to
batch (pad, collate, etc.) for their pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch

from viterbo.math.constructions import rotated_regular_ngon2d
from viterbo.math.polytope import vertices_to_halfspaces


class AtlasTinyRaggedRow(TypedDict):
    """AtlasTiny row before derived quantities are attached."""

    polytope_id: str
    generator: str
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor


class AtlasTinyRow(TypedDict):
    """Completed AtlasTiny row with derived quantities attached."""

    polytope_id: str
    generator: str
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor
    volume: torch.Tensor
    capacity_ehz: torch.Tensor | None
    systolic_ratio: torch.Tensor | None
    minimal_action_cycle: torch.Tensor | None
    num_vertices: int
    num_facets: int
    dimension: int


def atlas_tiny_generate() -> list[AtlasTinyRaggedRow]:
    """Generate deterministic polytopes suitable for symplectic evaluation."""

    dtype = torch.float64
    base_specs: Sequence[tuple[str, str, int, float, float]] = (
        ("sq_unit", "regular_ngon", 4, 0.0, 1.0),
        ("pentagon_rot", "regular_ngon", 5, 0.35, 1.0),
        ("hexagon_scaled", "regular_ngon_scaled", 6, -0.2, 1.5),
    )

    rows: list[AtlasTinyRaggedRow] = []
    for polytope_id, generator_name, sides, angle, scale in base_specs:
        vertices, normals, offsets = rotated_regular_ngon2d(sides, angle)
        vertices = vertices * scale
        normals, offsets = vertices_to_halfspaces(vertices)
        rows.append(
            {
                "polytope_id": polytope_id,
                "generator": generator_name,
                "vertices": vertices.to(dtype=dtype),
                "normals": normals.to(dtype=dtype),
                "offsets": offsets.to(dtype=dtype),
            }
        )
    return rows


def atlas_tiny_complete_row(row: AtlasTinyRaggedRow) -> AtlasTinyRow:
    """Populate derived quantities for a ragged row using math utilities."""

    from viterbo.math.capacity_ehz.cycle import minimal_action_cycle
    from viterbo.math.capacity_ehz.ratios import systolic_ratio
    from viterbo.math.volume import volume as volume_from_vertices

    vertices = row["vertices"]
    normals = row["normals"]
    offsets = row["offsets"]

    if vertices.ndim != 2:
        raise ValueError("vertices must be (M, D) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,) tensor")
    if vertices.device != normals.device or vertices.device != offsets.device:
        raise ValueError("vertices, normals, and offsets must share the same device")
    if vertices.dtype != normals.dtype or vertices.dtype != offsets.dtype:
        raise ValueError("vertices, normals, and offsets must share the same dtype")

    dim = int(vertices.size(1))
    volume = volume_from_vertices(vertices)
    capacity: torch.Tensor | None = None
    cycle: torch.Tensor | None = None
    if dim == 2:
        capacity, cycle = minimal_action_cycle(vertices, normals, offsets)

    systolic: torch.Tensor | None = None
    if capacity is not None:
        systolic = systolic_ratio(volume, capacity, dim)

    return AtlasTinyRow(
        polytope_id=row["polytope_id"],
        generator=row["generator"],
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        capacity_ehz=capacity,
        systolic_ratio=systolic,
        minimal_action_cycle=cycle,
        num_vertices=int(vertices.size(0)),
        num_facets=int(normals.size(0)),
        dimension=dim,
    )


def atlas_tiny_build() -> list[AtlasTinyRow]:
    """Return completed AtlasTiny rows as a list of typed dictionaries."""

    rows = atlas_tiny_generate()
    return [atlas_tiny_complete_row(row) for row in rows]


def atlas_tiny_collate_pad(rows: Sequence[AtlasTinyRow]) -> dict[str, torch.Tensor | list[str]]:
    """Pad a batch of AtlasTiny rows to the maximum vertex/facet counts.

    Args:
      rows: sequence of completed AtlasTiny rows.

    Returns:
      Dictionary with padded tensors:
        - ``polytope_id``: list[str]
        - ``generator``: list[str]
        - ``vertices``: (B, V_max, D)
        - ``normals``: (B, F_max, D)
        - ``offsets``: (B, F_max)
        - ``minimal_action_cycle``: (B, C_max, D)
        - ``vertex_mask``: (B, V_max) bool
        - ``facet_mask``: (B, F_max) bool
        - ``cycle_mask``: (B, C_max) bool
        - scalars ``volume``, ``capacity_ehz``, ``systolic_ratio`` of shape (B,)
    """

    if not rows:
        raise ValueError("atlas_tiny_collate_pad requires a non-empty batch")

    dtype = rows[0]["vertices"].dtype
    device = rows[0]["vertices"].device
    dim = rows[0]["vertices"].size(1)

    max_vertices = max(row["vertices"].size(0) for row in rows)
    max_facets = max(row["normals"].size(0) for row in rows)
    max_cycle = max(
        row["minimal_action_cycle"].size(0) if row["minimal_action_cycle"] is not None else 0
        for row in rows
    )

    batch_size = len(rows)
    vertices = torch.zeros((batch_size, max_vertices, dim), dtype=dtype, device=device)
    normals = torch.zeros((batch_size, max_facets, dim), dtype=dtype, device=device)
    offsets = torch.zeros((batch_size, max_facets), dtype=dtype, device=device)
    cycle = torch.zeros((batch_size, max_cycle, dim), dtype=dtype, device=device)

    vertex_mask = torch.zeros((batch_size, max_vertices), dtype=torch.bool, device=device)
    facet_mask = torch.zeros((batch_size, max_facets), dtype=torch.bool, device=device)
    cycle_mask = torch.zeros((batch_size, max_cycle), dtype=torch.bool, device=device)

    volume = torch.zeros((batch_size,), dtype=dtype, device=device)
    capacity = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
    systolic = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)

    polytope_ids: list[str] = []
    generators: list[str] = []

    for i, row in enumerate(rows):
        polytope_ids.append(row["polytope_id"])
        generators.append(row["generator"])

        v = row["vertices"]
        n = row["normals"]
        o = row["offsets"]
        vertices[i, : v.size(0)] = v
        normals[i, : n.size(0)] = n
        offsets[i, : o.size(0)] = o
        vertex_mask[i, : v.size(0)] = True
        facet_mask[i, : n.size(0)] = True

        if row["minimal_action_cycle"] is not None and row["minimal_action_cycle"].size(0) > 0:
            c = row["minimal_action_cycle"]
            cycle[i, : c.size(0)] = c
            cycle_mask[i, : c.size(0)] = True

        volume[i] = row["volume"]
        if row["capacity_ehz"] is not None:
            capacity[i] = row["capacity_ehz"]
        if row["systolic_ratio"] is not None:
            systolic[i] = row["systolic_ratio"]

    return {
        "polytope_id": polytope_ids,
        "generator": generators,
        "vertices": vertices,
        "normals": normals,
        "offsets": offsets,
        "minimal_action_cycle": cycle,
        "vertex_mask": vertex_mask,
        "facet_mask": facet_mask,
        "cycle_mask": cycle_mask,
        "volume": volume,
        "capacity_ehz": capacity,
        "systolic_ratio": systolic,
    }
