"""Geometry of a fixed convex polytope (representations and queries).

This module focuses on per-body geometry:
- H/V representations and conversions
- basic queries (support, distances, bounding boxes, halfspace violations)

All functions are pure and Torch-first (preserve dtype/device; no implicit moves).
"""

from __future__ import annotations

import itertools

import torch
from scipy.spatial import ConvexHull

# ---- Basic queries -----------------------------------------------------------


def support(points: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Support function of a finite point set.

    Args:
      points: (N, D) float tensor.
      direction: (D,) float tensor. Not required to be normalized.

    Returns:
      Scalar tensor: max_i <points[i], direction>.
    """
    return (points @ direction).max()


def support_argmax(points: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Support value and index of the maximiser.

    Args:
      points: (N, D) float tensor.
      direction: (D,) float tensor.

    Returns:
      (value, index) where value is scalar tensor and index is Python int.
    """
    vals = points @ direction
    val, idx = torch.max(vals, dim=0)
    return val, int(idx.item())


def pairwise_squared_distances(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances.

    Args:
      points: (N, D) float tensor.

    Returns:
      (N, N) float tensor of squared distances.
    """
    x2 = (points * points).sum(dim=1, keepdim=True)
    y2 = x2.transpose(0, 1)
    xy = points @ points.T
    d2 = x2 + y2 - 2.0 * xy
    return d2.clamp_min_(0.0)


def halfspace_violations(
    points: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Compute positive violations of halfspaces for each point.

    Given halfspaces Bx <= c (normals = B, offsets = c), returns relu(Bx - c).
    """
    bx = points @ normals.T  # (N, F)
    c = offsets.unsqueeze(0)  # (1, F)
    viol = bx - c
    return torch.clamp_min(viol, 0.0)


def bounding_box(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned bounding box (min, max) for a point cloud."""
    return points.min(dim=0).values, points.max(dim=0).values


# ---- H/V representations -----------------------------------------------------


def _ensure_full_dimension(vertices: torch.Tensor) -> None:
    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D tensor")
    if vertices.size(0) <= vertices.size(1):
        raise ValueError("need at least D + 1 vertices for a full-dimensional hull")
    rank = torch.linalg.matrix_rank(vertices - vertices.mean(dim=0, keepdim=True))
    if int(rank.item()) != vertices.size(1):
        raise ValueError("vertices must span a full-dimensional polytope")


def _lexicographic_order(points: torch.Tensor) -> torch.Tensor:
    order = torch.arange(points.size(0))
    for dim in range(points.size(1) - 1, -1, -1):
        values = points[:, dim]
        order = order[torch.argsort(values[order])]
    return order


def _pairwise_unique(
    normals: list[torch.Tensor], offsets: list[torch.Tensor], tol: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    uniq_normals: list[torch.Tensor] = []
    uniq_offsets: list[torch.Tensor] = []
    for normal, offset in zip(normals, offsets):
        duplicate = False
        for existing_normal, existing_offset in zip(uniq_normals, uniq_offsets):
            if torch.allclose(normal, existing_normal, atol=tol, rtol=0.0) and torch.allclose(
                offset, existing_offset, atol=tol, rtol=0.0
            ):
                duplicate = True
                break
        if not duplicate:
            uniq_normals.append(normal)
            uniq_offsets.append(offset)
    return uniq_normals, uniq_offsets


def _facet_from_indices(
    vertices: torch.Tensor, indices: tuple[int, ...], centroid: torch.Tensor, tol: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    subset = vertices[list(indices)]
    base = subset[0]
    diffs = subset[1:] - base
    if diffs.size(0) == 0:
        return None
    rank = torch.linalg.matrix_rank(diffs)
    if int(rank.item()) != diffs.size(0):
        return None
    _, _, vh = torch.linalg.svd(diffs, full_matrices=True)
    normal = vh[-1]
    norm = torch.linalg.norm(normal)
    if norm <= tol:
        return None
    normal = normal / norm
    offset = torch.dot(normal, base)
    if (centroid @ normal) - offset > tol:
        normal = -normal
        offset = -offset
    support_values = vertices @ normal
    if torch.max(support_values - offset) > tol:
        return None
    if torch.sum(torch.isclose(support_values, offset, atol=tol, rtol=0.0)) < vertices.size(1):
        return None
    return normal, offset


def vertices_to_halfspaces(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert vertices to half-space representation ``Bx <= c``.

    Returns ``(normals, offsets)`` with shapes ``(F, D)`` and ``(F,)``.
    """
    _ensure_full_dimension(vertices)
    dtype = vertices.dtype
    device = vertices.device
    dim = vertices.size(1)
    tol = max(float(torch.finfo(dtype).eps) ** 0.5, 1e-9)

    if dim == 1:
        min_val = vertices.min()
        max_val = vertices.max()
        normals = torch.tensor([[1.0], [-1.0]], dtype=dtype, device=device)
        offsets = torch.stack([max_val, -min_val])
        return normals, offsets

    vertices_cpu = vertices.detach().to(dtype=torch.float64, device="cpu")
    hull = ConvexHull(vertices_cpu.numpy())
    normals_np = hull.equations[:, :-1]
    offsets_np = -hull.equations[:, -1]
    norms = torch.from_numpy((normals_np**2).sum(axis=1, keepdims=True)).sqrt()
    normals_tensor = torch.from_numpy(normals_np) / norms
    offsets_tensor = torch.from_numpy(offsets_np) / norms.squeeze(1)
    normals_list = [n.clone() for n in normals_tensor]
    offsets_list = [o.clone() for o in offsets_tensor]
    unique_normals, unique_offsets = _pairwise_unique(normals_list, offsets_list, tol)
    normals_final = torch.stack(unique_normals)
    offsets_final = torch.stack(unique_offsets)
    return normals_final.to(device=device, dtype=dtype), offsets_final.to(device=device, dtype=dtype)


def halfspaces_to_vertices(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Convert half-space representation to vertices.

    Returns vertices ``(M, D)`` in lexicographic order.
    """
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must have matching first dimension")
    dim = normals.size(1)
    if dim == 0:
        raise ValueError("dimension must be positive")
    tol = max(float(torch.finfo(normals.dtype).eps) ** 0.5, 1e-9)
    candidates: list[torch.Tensor] = []
    for indices in itertools.combinations(range(normals.size(0)), dim):
        sub_normals = normals[list(indices)]
        sub_offsets = offsets[list(indices)]
        try:
            vertex = torch.linalg.solve(sub_normals, sub_offsets)
        except RuntimeError:
            continue
        if torch.max(normals @ vertex - offsets) > tol:
            continue
        if any(torch.allclose(vertex, existing, atol=tol, rtol=0.0) for existing in candidates):
            continue
        candidates.append(vertex)
    if not candidates:
        raise ValueError("no feasible vertices found for the provided halfspaces")
    vertices_tensor = torch.stack(candidates)
    cpu_vertices = vertices_tensor.detach().cpu()
    order = _lexicographic_order(cpu_vertices)
    ordered = vertices_tensor[order.to(vertices_tensor.device)]
    return ordered


# ---- Incidence/adjacency (stubs) --------------------------------------------


def facet_vertex_incidence(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Return a boolean incidence matrix of shape ``(F, M)`` (stub)."""
    raise NotImplementedError
