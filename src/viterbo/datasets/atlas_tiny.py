"""AtlasTiny dataset scaffolding (stubs).

Defines a small synthetic dataset of polytopes with geometric/symplectic
attributes. This module is an adapter layer around `viterbo.math` utilities;
it should avoid heavy dependencies at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


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

    from viterbo.math.geometry import volume as volume_from_vertices
    from viterbo.math.symplectic import minimal_action_cycle, systolic_ratio

    out = AtlasTinyRow(**{k: v for k, v in row.__dict__.items()})
    if out.volume is None:
        out.volume = volume_from_vertices(out.vertices)
    if out.minimal_action_cycle is None or out.capacity_ehz is None:
        capacity, cycle = minimal_action_cycle(out.vertices, out.normals, out.offsets)
        out.capacity_ehz = capacity
        out.minimal_action_cycle = cycle
    if out.systolic_ratio is None:
        assert out.volume is not None and out.capacity_ehz is not None
        out.systolic_ratio = systolic_ratio(out.volume, out.capacity_ehz)
    return out


def atlas_tiny_generate() -> list[AtlasTinyRow]:
    """Generate a list of rows with geometric data (stubs)."""

    raise NotImplementedError


class AtlasTinyDataset(Dataset[AtlasTinyRow]):
    """Torch dataset wrapping completed AtlasTiny rows."""

    def __init__(self, rows: list[AtlasTinyRow]) -> None:
        self._rows = rows

    def __len__(self) -> int:  # noqa: D401 - brief
        return len(self._rows)

    def __getitem__(self, idx: int) -> AtlasTinyRow:  # noqa: D401 - brief
        return self._rows[idx]


def atlas_tiny_build() -> AtlasTinyDataset:
    """Build and return a `torch.utils.data.Dataset` of completed rows."""

    rows = atlas_tiny_generate()
    complete_rows = [atlas_tiny_complete_row(row) for row in rows]
    return AtlasTinyDataset(complete_rows)
