"""Derived ratios and scalars built from capacities."""

from __future__ import annotations

import torch


def systolic_ratio(
    volume: torch.Tensor, capacity_ehz: torch.Tensor, symplectic_dimension: int | None = None
) -> torch.Tensor:
    r"""Viterbo systolic ratio ``vol(K) / c_{EHZ}(K)^{n}`` for ``2n``-dimensional bodies."""
    if volume.ndim != 0 or capacity_ehz.ndim != 0:
        raise ValueError("volume and capacity_ehz must be scalar tensors")
    if torch.any(capacity_ehz <= 0):
        raise ValueError("capacity_ehz must be strictly positive")
    if symplectic_dimension is None:
        raise ValueError("symplectic_dimension must be provided for systolic_ratio")
    if symplectic_dimension % 2 != 0 or symplectic_dimension <= 0:
        raise ValueError("symplectic_dimension must be a positive even integer")
    n = symplectic_dimension // 2
    return volume / capacity_ehz.pow(n)
