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
        k = int(
            torch.randint(self.min_points, self.max_points + 1, (1,), generator=self._rng).item()
        )
        points = torch.randn(
            (k, self.dim), generator=self._rng, device=self.device, dtype=self.dtype
        )
        direction = torch.randn(
            (self.dim,), generator=self._rng, device=self.device, dtype=self.dtype
        )
        return Sample(points=points, direction=direction)


def _validate_ragged_batch(
    batch: list[Sample], *, caller: str
) -> tuple[int, torch.device, torch.dtype, list[int]]:
    """Ensure a ragged batch is well-formed for downstream collators."""

    if not batch:
        raise ValueError(f"{caller} expected a non-empty batch of samples")

    first = batch[0]
    if not isinstance(first, Sample):
        raise ValueError(f"{caller} expected Sample instances, but received {type(first).__name__}")

    points = first.points
    direction = first.direction

    if points.ndim != 2:
        raise ValueError(f"{caller} requires points tensors with shape (K, D); got {points.shape}")
    if direction.ndim != 1:
        raise ValueError(
            f"{caller} requires direction tensors with shape (D,); got {direction.shape}"
        )

    dim = points.shape[1]
    if direction.shape[0] != dim:
        raise ValueError(
            f"{caller} requires matching direction dimension {dim}; got {direction.shape[0]}"
        )

    device = points.device
    dtype = points.dtype

    if direction.device != device:
        raise ValueError(
            f"{caller} requires points and directions on the same device; got {device} and {direction.device}"
        )
    if direction.dtype != dtype:
        raise ValueError(
            f"{caller} requires points and directions with the same dtype; got {dtype} and {direction.dtype}"
        )

    lengths: list[int] = []
    for i, sample in enumerate(batch):
        if not isinstance(sample, Sample):
            raise ValueError(
                f"{caller} expected Sample instances, but received {type(sample).__name__} at index {i}"
            )

        pts = sample.points
        drn = sample.direction

        if pts.ndim != 2:
            raise ValueError(
                f"{caller} requires points tensors with shape (K, D); got {pts.shape} at index {i}"
            )
        if drn.ndim != 1:
            raise ValueError(
                f"{caller} requires direction tensors with shape (D,); got {drn.shape} at index {i}"
            )

        if pts.shape[1] != dim:
            raise ValueError(
                f"{caller} expected all points tensors to have D={dim}; got {pts.shape[1]} at index {i}"
            )
        if drn.shape[0] != dim:
            raise ValueError(
                f"{caller} expected all direction tensors to have length {dim}; got {drn.shape[0]} at index {i}"
            )

        if pts.device != device or drn.device != device:
            raise ValueError(
                f"{caller} requires a single device per batch; sample {i} uses {pts.device} / {drn.device} instead of {device}"
            )
        if pts.dtype != dtype or drn.dtype != dtype:
            raise ValueError(
                f"{caller} requires a single dtype per batch; sample {i} uses {pts.dtype} / {drn.dtype} instead of {dtype}"
            )

        lengths.append(int(pts.shape[0]))

    return dim, device, dtype, lengths


def collate_list(batch: list[Sample]) -> dict[str, list[torch.Tensor]]:
    """Collate into lists of variable-length tensors (no padding).

    Returns a dict with keys `points` (list[(K_i, D)]) and `direction` (list[(D,)]).

    Raises:
      ValueError: if the batch is empty, or tensors differ in shape, device, or dtype.
    """

    _validate_ragged_batch(batch, caller="collate_list")
    points_list = [sample.points for sample in batch]
    direction_list = [sample.direction for sample in batch]
    return {"points": points_list, "direction": direction_list}


def collate_pad(batch: list[Sample]) -> dict[str, torch.Tensor]:
    """Collate with right-padding to the max K in batch and a boolean mask.

    Returns:
      points: (B, K_max, D) padded with zeros (float dtype of inputs).
      mask: (B, K_max) boolean mask with True for valid entries.
      direction: (B, D)

    Raises:
      ValueError: if the batch is empty, or tensors differ in shape, device, or dtype.
    """

    dim, device, dtype, lengths = _validate_ragged_batch(batch, caller="collate_pad")

    bsz = len(batch)
    k_max = max(lengths)
    points = torch.zeros((bsz, k_max, dim), device=device, dtype=dtype)
    mask = torch.zeros((bsz, k_max), device=device, dtype=torch.bool)
    direction = torch.stack([sample.direction for sample in batch], dim=0)

    for i, sample in enumerate(batch):
        k = lengths[i]
        if k == 0:
            continue
        points[i, :k] = sample.points
        mask[i, :k] = True

    return {"points": points, "mask": mask, "direction": direction}
