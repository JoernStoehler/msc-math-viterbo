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

