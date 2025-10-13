"""Dummy generator: write a small dataset artefact.

Usage:
  uv run python notebooks/dummy_generate.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from viterbo.datasets import RaggedPointsDataset, collate_pad


def main() -> None:
    artefacts = Path("artefacts")
    artefacts.mkdir(parents=True, exist_ok=True)

    ds = RaggedPointsDataset(num_samples=16, dim=4, min_points=3, max_points=9, seed=42)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_pad)
    batch = next(iter(loader))
    out = {"points": batch["points"], "mask": batch["mask"], "direction": batch["direction"]}
    path = artefacts / "dummy_dataset.pt"
    torch.save(out, path)
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

