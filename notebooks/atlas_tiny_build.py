# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AtlasTiny v1 — Build and Save (Parquet/HF)
#
# Build the AtlasTiny v1 rows in-memory, convert to a Hugging Face Dataset, and
# save as a Parquet artefact under `artefacts/datasets/atlas-tiny/v1/`.
#
# This notebook is runnable as a script via Jupytext:
#   uv run python notebooks/atlas_tiny_build.py

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Ascend from start to locate the repo root by pyproject.toml."""
    cur = start
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


try:
    PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
except NameError:
    PROJECT_ROOT = _find_project_root(Path.cwd().resolve())
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from viterbo.datasets.atlas_tiny import atlas_tiny_build
from viterbo.datasets.atlas_tiny_io import (
    atlas_tiny_save_parquet,
    atlas_tiny_to_hf,
)


# %% [markdown]
# ## Build rows and save Parquet


# %%
def main(out_dir: Path | None = None) -> None:
    if out_dir is None:
        out_dir = PROJECT_ROOT / "artefacts" / "datasets" / "atlas-tiny" / "v1"
    print(f"Building AtlasTiny v1 rows (output dir: {out_dir}) ...")
    rows = atlas_tiny_build()
    print(f"Built {len(rows)} rows. Converting to HF Dataset ...")
    ds = atlas_tiny_to_hf(rows)
    print("Saving Parquet and companion metadata ...")
    atlas_tiny_save_parquet(ds, os.fspath(out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
