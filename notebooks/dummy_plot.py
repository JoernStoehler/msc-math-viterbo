"""Dummy consumer: load artefact and produce a plot.

Usage:
  uv run python notebooks/dummy_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    artefacts = Path("artefacts")
    path = artefacts / "dummy_dataset.pt"
    data = torch.load(path)
    # Plot histogram of valid point counts per sample
    mask = data.get("mask")
    if mask is None:
        print("No mask found; nothing to plot.")
        return
    counts = mask.sum(dim=1).cpu().numpy()
    plt.figure(figsize=(4, 3))
    plt.hist(counts, bins=range(int(counts.min()), int(counts.max()) + 2))
    plt.title("Valid points per sample")
    plt.xlabel("K")
    plt.ylabel("Frequency")
    artefacts.mkdir(parents=True, exist_ok=True)
    out = artefacts / "dummy_plot.png"
    plt.tight_layout()
    plt.savefig(out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

