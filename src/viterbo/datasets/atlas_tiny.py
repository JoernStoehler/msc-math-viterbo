"""AtlasTiny dataset scaffolding (stubs).

Defines a small synthetic dataset of polytopes with geometric/symplectic
attributes. This module is an adapter layer around `viterbo.math` utilities;
it should avoid heavy dependencies at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from viterbo.math.constructions import rotated_regular_ngon2d
from viterbo.math.polytope import vertices_to_halfspaces

@dataclass
class AtlasTinyRow:
    """Row schema for AtlasTiny.

    Attributes:
      polytope_id: unique identifier.
      generator: name of the generator method.
      vertices: (M, D) float tensor.
      normals: (F, D) float tensor.
      offsets: (F,) float tensor.
      volume: () float tensor.
      capacity_ehz: () float tensor.
      systolic_ratio: () float tensor.
      minimal_action_cycle: (K, D) float tensor points of cycle.
    """

    polytope_id: str
    generator: str
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor
    volume: torch.Tensor | None = None
    capacity_ehz: torch.Tensor | None = None
    systolic_ratio: torch.Tensor | None = None
    minimal_action_cycle: torch.Tensor | None = None


def atlas_tiny_complete_row(row: AtlasTinyRow) -> AtlasTinyRow:
    """Compute missing attributes for a row using math utilities.

    This function imports from ``viterbo.math`` lazily to avoid import-time cost.
    """

    from viterbo.math.minimal_action import minimal_action_cycle, systolic_ratio
    from viterbo.math.volume import volume as volume_from_vertices

    out = AtlasTinyRow(**{k: v for k, v in row.__dict__.items()})
    dim = int(out.vertices.size(1))
    # Only compute derived quantities for dimensions currently supported by the math layer.
    # Keep 4D focus by not attempting unsupported computations.
    if dim == 1 or dim == 2 or dim == 3:
        if out.volume is None:
            out.volume = volume_from_vertices(out.vertices)
    if dim == 2:
        if out.minimal_action_cycle is None or out.capacity_ehz is None:
            capacity, cycle = minimal_action_cycle(out.vertices, out.normals, out.offsets)
            out.capacity_ehz = capacity
            out.minimal_action_cycle = cycle
    if out.systolic_ratio is None and out.volume is not None and out.capacity_ehz is not None:
        out.systolic_ratio = systolic_ratio(out.volume, out.capacity_ehz, dim)
    return out


def atlas_tiny_generate() -> list[AtlasTinyRow]:
    """Generate a deterministic list of low-dimensional symplectic polytopes.

    The current focus is 2D polygons so the symplectic helpers (capacity, minimal
    action) can run end-to-end. Each row specifies vertices plus an H-representation;
    derived quantities are filled by :func:`atlas_tiny_complete_row`.
    """

    def _regular_ngon_row(
        *,
        polytope_id: str,
        generator: str,
        sides: int,
        angle: float,
        scale: float = 1.0,
    ) -> AtlasTinyRow:
        vertices, normals, offsets = rotated_regular_ngon2d(sides, angle)
        if scale != 1.0:
            vertices = vertices * scale
            normals, offsets = vertices_to_halfspaces(vertices)
        dtype = torch.float64
        return AtlasTinyRow(
            polytope_id=polytope_id,
            generator=generator,
            vertices=vertices.to(dtype=dtype),
            normals=normals.to(dtype=dtype),
            offsets=offsets.to(dtype=dtype),
        )

    rows = [
        _regular_ngon_row(
            polytope_id="sq_unit",
            generator="regular_ngon",
            sides=4,
            angle=0.0,
        ),
        _regular_ngon_row(
            polytope_id="pentagon_rot",
            generator="regular_ngon",
            sides=5,
            angle=0.35,
        ),
        _regular_ngon_row(
            polytope_id="hexagon_scaled",
            generator="regular_ngon_scaled",
            sides=6,
            angle=-0.2,
            scale=1.5,
        ),
    ]

    return rows


class AtlasTinyDataset(Dataset[AtlasTinyRow]):
    """Torch dataset wrapping completed AtlasTiny rows."""

    def __init__(self, rows: list[AtlasTinyRow]) -> None:
        """Store precomputed rows.

        Args:
          rows: Completed rows to expose via the dataset interface.
        """
        self._rows = rows

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)

    def __getitem__(self, idx: int) -> AtlasTinyRow:
        """Return the row at index ``idx``."""
        return self._rows[idx]


def atlas_tiny_build() -> AtlasTinyDataset:
    """Build and return a `torch.utils.data.Dataset` of completed rows."""

    rows = atlas_tiny_generate()
    complete_rows = [atlas_tiny_complete_row(row) for row in rows]
    return AtlasTinyDataset(complete_rows)
