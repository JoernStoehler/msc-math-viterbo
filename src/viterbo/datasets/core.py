from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    """Container for a single ragged sample.

    Attributes:
      points: (K, D) float tensor for K points in D dims.
      direction: (D,) float tensor; can be used with geometry.support.
    """

    points: torch.Tensor
    direction: torch.Tensor


class RaggedPointsDataset(Dataset[Sample]):
    """Generate a simple ragged dataset of point sets and directions.

    Args:
      num_samples: number of samples.
      dim: ambient dimension.
      min_points: min number of points per sample.
      max_points: max number of points per sample (inclusive).
      seed: integer seed for reproducibility (optional).
      device: optional torch device for generated tensors.
      dtype: torch dtype (default float32).
    """

    def __init__(
        self,
        num_samples: int,
        dim: int,
        min_points: int = 3,
        max_points: int = 12,
        seed: int | None = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize dataset parameters and RNG."""
        assert min_points >= 1 and max_points >= min_points
        self.num_samples = int(num_samples)
        self.dim = int(dim)
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.device = device
        self.dtype = dtype
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(int(seed))

    def __len__(self) -> int:
        """Number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Sample:
        """Generate a single sample with a random number of points."""
        k = int(torch.randint(self.min_points, self.max_points + 1, (1,), generator=self._rng).item())
        points = torch.randn((k, self.dim), generator=self._rng, device=self.device, dtype=self.dtype)
        direction = torch.randn((self.dim,), generator=self._rng, device=self.device, dtype=self.dtype)
        return Sample(points=points, direction=direction)


def collate_list(batch: list[Sample]) -> dict[str, list[torch.Tensor]]:
    """Collate into lists of variable-length tensors (no padding).

    Returns a dict with keys `points` (list[(K_i, D)]) and `direction` (list[(D,)]).
    """

    points_list = [b.points for b in batch]
    direction_list = [b.direction for b in batch]
    return {"points": points_list, "direction": direction_list}


def collate_pad(batch: list[Sample]) -> dict[str, torch.Tensor]:
    """Collate with right-padding to the max K in batch and a boolean mask.

    Returns:
      points: (B, K_max, D) padded with zeros.
      mask: (B, K_max) True for valid entries.
      direction: (B, D)
    """

    bsz = len(batch)
    k_max = max(b.points.shape[0] for b in batch)
    dim = batch[0].points.shape[1]
    device = batch[0].points.device
    dtype = batch[0].points.dtype
    points = torch.zeros((bsz, k_max, dim), device=device, dtype=dtype)
    mask = torch.zeros((bsz, k_max), device=device, dtype=torch.bool)
    direction = torch.stack([b.direction for b in batch], dim=0)
    for i, b in enumerate(batch):
        k = b.points.shape[0]
        points[i, :k] = b.points
        mask[i, :k] = True
    return {"points": points, "mask": mask, "direction": direction}
