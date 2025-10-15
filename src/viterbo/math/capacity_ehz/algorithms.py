"""EHZ capacity solvers (entry points and simple variants)."""

from __future__ import annotations

import torch

from viterbo.math.capacity_ehz.common import polygon_area, split_lagrangian_product_vertices
from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product
from viterbo.math.polytope import halfspaces_to_vertices, vertices_to_halfspaces


def capacity_ehz_algorithm1(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""Ekeland–Hofer–Zehnder capacity via the Artstein–Avidan–Ostrover program (2D placeholder)."""
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must share the first dimension")
    d = normals.size(1)
    if d == 2:
        vertices = halfspaces_to_vertices(normals, offsets)
        return polygon_area(vertices)
    if d == 4:
        vertices = halfspaces_to_vertices(normals, offsets)
        return capacity_ehz_algorithm2(vertices)
    raise NotImplementedError(
        "capacity_ehz_algorithm1 currently supports d=2 or Lagrangian products in d=4"
    )


def capacity_ehz_algorithm2(vertices: torch.Tensor) -> torch.Tensor:
    r"""EHZ capacity via discrete billiards on vertices (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    d = vertices.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d == 4:
        tol = max(float(torch.finfo(vertices.dtype).eps) ** 0.5, 1e-9)
        vertices_q, vertices_p = split_lagrangian_product_vertices(vertices, tol)
        normals_p, offsets_p = vertices_to_halfspaces(vertices_p)
        capacity, _ = minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p)
        return capacity.to(dtype=vertices.dtype, device=vertices.device)
    if d != 2:
        raise NotImplementedError(
            "capacity_ehz_algorithm2 currently supports 2D and certain 4D inputs only"
        )
    if vertices.size(0) < 3:
        raise ValueError("need at least three vertices for a 2D polygon")
    return polygon_area(vertices)


def capacity_ehz_primal_dual(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    r"""Hybrid primal–dual EHZ capacity solver (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(1) != vertices.size(1):
        raise ValueError("vertices and normals must share the same ambient dimension")
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d == 2:
        capacity = capacity_ehz_algorithm2(vertices)
        reference = capacity_ehz_algorithm1(normals, offsets)
        if not torch.allclose(capacity, reference, atol=1e-8, rtol=1e-8):
            raise ValueError("inconsistent primal and dual capacities for the provided polygon")
        return capacity
    if d == 4:
        capacity = capacity_ehz_algorithm2(vertices)
        reference = capacity_ehz_algorithm1(normals, offsets)
        if not torch.allclose(capacity, reference, atol=1e-8, rtol=1e-8):
            raise ValueError("inconsistent primal and dual capacities for the provided polytope")
        return capacity
    raise NotImplementedError(
        "capacity_ehz_primal_dual currently supports 2D and certain 4D inputs only"
    )
