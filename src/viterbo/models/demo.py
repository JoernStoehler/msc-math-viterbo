from __future__ import annotations

from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader

from viterbo.math.geometry import support


def run_probe(
    loader: DataLoader,
    device: torch.device | None = None,
    max_batches: int | None = 2,
) -> Dict[str, float]:
    """Run a tiny probe that computes a support statistic over a few batches.

    Args:
      loader: DataLoader producing dicts with keys `points` and `direction`.
      device: Optional device to move data for compute (e.g., cuda). If None, infer from batch.
      max_batches: Maximal number of batches to consume.

    Returns:
      Dict with a single metric `avg_support`.
    """
    total = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch["points"], list):
            # Ragged list collate
            for pts, direction in zip(batch["points"], batch["direction"]):
                dev = device or pts.device
                s = support(pts.to(dev), direction.to(dev))
                total += float(s.detach().cpu())
                count += 1
        else:
            # Padded collate
            pts = batch["points"]
            mask = batch.get("mask")
            direction = batch["direction"]
            dev = device or pts.device
            if mask is None:
                # Treat as dense: iterate samples
                for p, d in zip(pts, direction):
                    s = support(p.to(dev), d.to(dev))
                    total += float(s.detach().cpu())
                    count += 1
            else:
                # Masked: iterate valid rows per sample
                for p, m, d in zip(pts, mask, direction):
                    k = int(m.sum().item())
                    s = support(p[:k].to(dev), d.to(dev))
                    total += float(s.detach().cpu())
                    count += 1
    avg = total / max(count, 1)
    return {"avg_support": avg}

