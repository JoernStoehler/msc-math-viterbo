"""Minimal-action Reeb orbit front-end and helpers.

Supported inputs
- 2D (planar): returns ``(area, CCW boundary cycle)`` where the area is the
  EHZ capacity surrogate for polygons, and the cycle is the input boundary
  ordered counter-clockwise.
- 4D: only certain Lagrangian product polytopes are supported. Detection uses
  ``split_lagrangian_product_vertices`` (cartesian-structured vertices) and the
  cycle is computed via the ``lagrangian_product`` path.

Other ambient dimensions are not implemented. The front-end is pure and does
not perform I/O; it preserves input dtype/device for outputs.
"""

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
    r"""Return minimal action and a representative periodic orbit.

    - 2D: returns ``(area, CCW cycle)``.
    - 4D: supports certain Lagrangian product polytopes only; otherwise raises
      ``NotImplementedError``.
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be (M, d) with d>=1")
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
        try:
            vertices_q, vertices_p = split_lagrangian_product_vertices(vertices, tol)
        except NotImplementedError as e:
            raise NotImplementedError(
                "minimal_action_cycle in 4D supports Lagrangian product polytopes only "
                "(cartesian vertex structure); input did not match"
            ) from e
        normals_p, offsets_p = vertices_to_halfspaces(vertices_p)
        capacity, cycle = minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p)
        return capacity.to(dtype=vertices.dtype, device=vertices.device), cycle.to(
            dtype=vertices.dtype, device=vertices.device
        )
    raise NotImplementedError(
        "minimal_action_cycle currently supports 2D and certain 4D inputs only"
    )
