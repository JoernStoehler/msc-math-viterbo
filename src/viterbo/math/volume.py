"""Polytope volume helpers and planned higher-dimensional backends.

Public API focuses on Torch-first, pure functions. Dtypes/devices follow inputs;
no implicit moves.
"""

from __future__ import annotations

import torch

from viterbo.math.polytope import vertices_to_halfspaces


def _convex_hull_order_2d(vertices: torch.Tensor) -> torch.Tensor:
    unique_vertices = torch.unique(vertices, dim=0)
    centre = unique_vertices.mean(dim=0)
    rel = unique_vertices - centre
    angles = torch.atan2(rel[:, 1], rel[:, 0])
    order = torch.argsort(angles)
    return unique_vertices[order]


def _facet_polygon_area(vertices: torch.Tensor, normal: torch.Tensor, tol: float) -> torch.Tensor:
    if vertices.size(0) < 3:
        return torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    centre = vertices.mean(dim=0)
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=vertices.dtype, device=vertices.device)
    if torch.abs(normal[0]) > 0.9:
        axis = torch.tensor([0.0, 1.0, 0.0], dtype=vertices.dtype, device=vertices.device)
    u = torch.cross(normal, axis, dim=0)
    if torch.linalg.norm(u) <= tol:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=vertices.dtype, device=vertices.device)
        u = torch.cross(normal, axis, dim=0)
    u = u / torch.linalg.norm(u)
    v = torch.cross(normal, u, dim=0)
    rel = vertices - centre
    coords = torch.stack((rel @ u, rel @ v), dim=1)
    ordered = _convex_hull_order_2d(coords)
    x = ordered[:, 0]
    y = ordered[:, 1]
    area = 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, shifts=-1)) - torch.sum(y * torch.roll(x, shifts=-1))
    )
    return area


def volume(vertices: torch.Tensor) -> torch.Tensor:
    """Volume of the convex hull for ``D ∈ {1, 2, 3}``.

    Plans for ``D ≥ 4`` include a triangulation backend, a Lawrence sign
    decomposition, and a quasi–Monte Carlo estimator; see the stubs below.
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be (M, D)")
    dim = vertices.size(1)
    if dim == 0:
        raise ValueError("dimension must be positive")
    dtype = vertices.dtype
    device = vertices.device
    tol = max(float(torch.finfo(dtype).eps) ** 0.5, 1e-9)
    if dim == 1:
        return (vertices.max(dim=0).values - vertices.min(dim=0).values).abs()[0]
    if dim == 2:
        ordered = _convex_hull_order_2d(vertices)
        x = ordered[:, 0]
        y = ordered[:, 1]
        area = 0.5 * torch.abs(
            torch.sum(x * torch.roll(y, shifts=-1)) - torch.sum(y * torch.roll(x, shifts=-1))
        )
        return area
    if dim != 3:
        raise NotImplementedError("volume currently implemented for dimensions 1, 2, and 3 only")

    normals, offsets = vertices_to_halfspaces(vertices)
    centre = vertices.mean(dim=0)
    vol = torch.zeros((), dtype=dtype, device=device)
    for normal, offset in zip(normals, offsets):
        mask = torch.isclose(vertices @ normal, offset, atol=tol)
        facet_vertices = vertices[mask]
        area = _facet_polygon_area(facet_vertices, normal, tol)
        height = offset - (centre @ normal)
        vol = vol + area * height / 3.0
    return torch.abs(vol)


def volume_via_triangulation(vertices: torch.Tensor) -> torch.Tensor:
    """Deterministic convex hull triangulation volume estimator (stub)."""
    raise NotImplementedError


def volume_via_lawrence(
    normals: torch.Tensor, offsets: torch.Tensor, *, basis: torch.Tensor | None = None
) -> torch.Tensor:
    """Lawrence sign decomposition volume from supporting halfspaces (stub)."""
    raise NotImplementedError


def volume_via_monte_carlo(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator | int,
) -> torch.Tensor:
    """Low-discrepancy Monte Carlo volume estimator for high dimensions (stub)."""
    raise NotImplementedError
