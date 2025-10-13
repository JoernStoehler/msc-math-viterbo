import torch


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


def matmul_vertices(matrix: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Apply linear map ``x -> A x`` to a vertex set.

    Args:
      matrix: (D, D) float tensor ``A``.
      vertices: (M, D) float tensor.

    Returns:
      new_vertices: (M, D) float tensor ``vertices @ A.T`` (implementation may differ).
    """
    raise NotImplementedError


def translate_vertices(translation: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Translate a vertex set by ``t``.

    Args:
      translation: (D,) float tensor ``t``.
      vertices: (M, D) float tensor.

    Returns:
      new_vertices: (M, D) float tensor of translated points.
    """
    raise NotImplementedError


def volume(vertices: torch.Tensor) -> torch.Tensor:
    """Volume of a convex polytope given its vertices.

    Intended as a reference implementation (e.g., via hull + simplices). Must be
    deterministic for a fixed input without hidden RNG.

    Args:
      vertices: (M, D) float tensor.

    Returns:
      volume: scalar float tensor.
    """
    raise NotImplementedError


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
    raise NotImplementedError
