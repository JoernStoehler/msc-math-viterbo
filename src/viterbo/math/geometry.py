from __future__ import annotations

import math

import torch

from .halfspaces import vertices_to_halfspaces


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
    # Using (x - y)^2 = x^2 + y^2 - 2xy trick for efficiency
    x2 = (points * points).sum(dim=1, keepdim=True)
    y2 = x2.transpose(0, 1)
    xy = points @ points.T
    d2 = x2 + y2 - 2.0 * xy
    return d2.clamp_min_(0.0)


def halfspace_violations(
    points: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Compute positive violations of halfspaces for each point.

    Given halfspaces Bx <= c (normals = B, offsets = c), returns
    relu(Bx - c) for all points.

    Args:
      points: (N, D)
      normals: (F, D)
      offsets: (F,)

    Returns:
      violations: (N, F) nonnegative.
    """
    bx = points @ normals.T  # (N, F)
    c = offsets.unsqueeze(0)  # (1, F)
    viol = bx - c
    return torch.clamp_min(viol, 0.0)


def bounding_box(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned bounding box (min, max) for a point cloud.

    Args:
      points: (N, D)

    Returns:
      (mins, maxs): each (D,)
    """
    return points.min(dim=0).values, points.max(dim=0).values


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
    area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1)) - torch.sum(y * torch.roll(x, shifts=-1)))
    return area


def matmul_vertices(matrix: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Apply linear map ``x -> A x`` to a vertex set.

    Args:
      matrix: (D, D) float tensor ``A``.
      vertices: (M, D) float tensor.

    Returns:
      new_vertices: (M, D) float tensor ``vertices @ A.T`` (implementation may differ).
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D")
    if matrix.size(0) != matrix.size(1) or matrix.size(1) != vertices.size(1):
        raise ValueError("matrix must be square with matching dimension to vertices")
    return vertices @ matrix.T


def translate_vertices(translation: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Translate a vertex set by ``t``.

    Args:
      translation: (D,) float tensor ``t``.
      vertices: (M, D) float tensor.

    Returns:
      new_vertices: (M, D) float tensor of translated points.
    """
    if translation.ndim != 1:
        raise ValueError("translation must be 1D tensor")
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D tensor")
    if translation.size(0) != vertices.size(1):
        raise ValueError("translation and vertices dimension mismatch")
    return vertices + translation


def volume(vertices: torch.Tensor) -> torch.Tensor:
    """Volume of a convex polytope given its vertices.

    Intended as a reference implementation (e.g., via hull + simplices). Must be
    deterministic for a fixed input without hidden RNG.

    Args:
      vertices: (M, D) float tensor.

    Returns:
      volume: scalar float tensor.
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
        area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1)) - torch.sum(y * torch.roll(x, shifts=-1)))
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


def rotated_regular_ngon2d(k: int, angle: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated regular n-gon in 2D.

    Args:
      k: number of sides ``k >= 3``.
      angle: rotation angle in radians.

    Returns:
      (vertices, normals, offsets):
        - vertices: (k, 2)
        - normals: (k, 2)
        - offsets: (k,)
    """
    if k < 3:
        raise ValueError("k must be at least 3")
    dtype = torch.get_default_dtype()
    device = torch.device("cpu")
    angles = (torch.arange(k, dtype=dtype, device=device) * (2.0 * math.pi / k)) + angle
    vertices = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    normals = []
    offsets = []
    for i in range(k):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % k]
        edge = v1 - v0
        normal = torch.tensor([edge[1], -edge[0]], dtype=dtype, device=device)
        normal = normal / torch.linalg.norm(normal)
        offset = torch.dot(normal, v0)
        if offset < 0:
            normal = -normal
            offset = -offset
        normals.append(normal)
        offsets.append(offset)
    normals_tensor = torch.stack(normals)
    offsets_tensor = torch.stack(offsets)
    return vertices, normals_tensor, offsets_tensor
