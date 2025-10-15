"""Minimal-action Reeb orbit front-end and helpers."""

from __future__ import annotations

import torch

from viterbo.math.capacity_ehz.common import (
    order_vertices_ccw,
    polygon_area,
    split_lagrangian_product_vertices,
)
from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product
from viterbo.math.polytope import vertices_to_halfspaces


def minimal_action_cycle(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the minimal action Reeb orbit (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if vertices.size(1) != normals.size(1):
        raise ValueError("vertices and normals must share the same ambient dimension")
    d = vertices.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d == 2:
        ordered_vertices = order_vertices_ccw(vertices)
        capacity = polygon_area(ordered_vertices)
        return capacity, ordered_vertices
    if d == 4:
        tol = max(float(torch.finfo(vertices.dtype).eps) ** 0.5, 1e-9)
        vertices_q, vertices_p = split_lagrangian_product_vertices(vertices, tol)
        normals_p, offsets_p = vertices_to_halfspaces(vertices_p)
        capacity, cycle = minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p)
        return capacity.to(dtype=vertices.dtype, device=vertices.device), cycle.to(
            dtype=vertices.dtype, device=vertices.device
        )
    raise NotImplementedError(
        "minimal_action_cycle currently supports 2D and certain 4D inputs only"
    )
