"""Polytope volume helpers with Torch-first, dimension-agnostic implementations."""

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


def _project_onto_affine_span(vertices: torch.Tensor, tol: float) -> torch.Tensor | None:
    """Project vertices to coordinates in their affine span."""

    unique_vertices = torch.unique(vertices, dim=0)
    if unique_vertices.size(0) <= 1:
        return None

    base = unique_vertices[0]
    rel = unique_vertices - base
    diffs = rel[1:]
    if diffs.size(0) == 0:
        return None
    u, s, vh = torch.linalg.svd(diffs, full_matrices=False)
    rank = int((s > tol).sum().item())
    if rank == 0:
        return None
    basis = vh[:rank]  # (rank, D)
    coords = rel @ basis.T  # (M, rank)
    return coords


def _facet_measure(vertices: torch.Tensor, tol: float) -> torch.Tensor:
    """Return the (d-1)-dimensional volume of a facet defined by ``vertices``."""

    coords = _project_onto_affine_span(vertices, tol)
    if coords is None:
        return torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    # Remove duplicate coordinates to improve stability in lower dimensions.
    coords = torch.unique(coords, dim=0)
    dim = coords.size(1)
    if dim == 0 or coords.size(0) <= 1:
        return torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    if coords.size(0) <= dim:
        return torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    return volume(coords)


def volume(vertices: torch.Tensor) -> torch.Tensor:
    """Volume of a convex polytope in ``R^D`` for ``D >= 1``."""
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

    normals, offsets = vertices_to_halfspaces(vertices)
    centre = vertices.mean(dim=0)
    vol = torch.zeros((), dtype=dtype, device=device)
    for normal, offset in zip(normals, offsets):
        mask = torch.isclose(vertices @ normal, offset, atol=tol, rtol=0.0)
        facet_vertices = vertices[mask]
        if facet_vertices.size(0) < dim:
            continue
        facet_measure = _facet_measure(facet_vertices, tol)
        height = offset - (centre @ normal)
        if height < 0:
            height = -height
        vol = vol + facet_measure * height / dim
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
