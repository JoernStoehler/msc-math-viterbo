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
# # AtlasTiny Explorer — Parquet/HF I/O
#
# A small, reproducible explorer for the AtlasTiny dataset. It loads rows from
# a Parquet artefact via Hugging Face Datasets helpers, reconstructs in-memory
# rows, shows a quick summary, builds a small padded batch, and creates a few
# plots.  The runtime stays short and CPU-only.
#
# Requirements and constraints:
# - Load using `viterbo.datasets.atlas_tiny_io.{atlas_tiny_load_parquet, atlas_tiny_rows_from_hf}`
# - Batch using `viterbo.datasets.atlas_tiny.atlas_tiny_collate_pad`
# - Respect float64 CPU tensors; do not import pyarrow/datasets directly
# - Save plots to `artefacts/published/atlas_tiny_explorer/` when run as a script

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch


# Ensure project `src/` is on the path for `viterbo.*` imports when run as a script/notebook.
def _find_project_root(start: Path) -> Path:
    cur = start
    for p in [cur] + list(cur.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start


try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = _find_project_root(Path.cwd().resolve())
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from viterbo.datasets.atlas_tiny import atlas_tiny_collate_pad


# %% [markdown]
# ## Configuration

# %%
# Parquet path (C1 generates this artefact). Accept either directory or the file itself.
PARQUET_PATH = PROJECT_ROOT / "artefacts" / "datasets" / "atlas-tiny" / "v1" / "data.parquet"

# Output directory for saved figures when not in a notebook environment.
PUBLISH_DIR = PROJECT_ROOT / "artefacts" / "published" / "atlas_tiny_explorer"
PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

SHOW_INLINE = False
try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None  # type: ignore[assignment]
else:
    SHOW_INLINE = get_ipython() is not None


# %% [markdown]
# ## Load dataset from Parquet via HF Datasets


# %%
def load_rows_from_parquet(path: Path) -> list[dict[str, Any]] | None:
    """Return reconstructed rows from Parquet, or None if artefact is missing.

    The helper handles both directory paths (containing `data.parquet`) and a
    direct Parquet file path.
    """

    # Gracefully handle missing artefacts (blocked on C1).
    parquet_path: Path
    if path.is_dir():
        parquet_path = path / "data.parquet"
    else:
        parquet_path = path

    if not parquet_path.exists():
        print(
            f"Parquet not found at '{parquet_path}'.\n"
            "This notebook is blocked by C1: AtlasTiny Parquet/HF I/O.\n"
            "Once C1 generates artefacts, re-run: uv run python notebooks/atlas_tiny_explorer.py"
        )
        return None

    # Import here to allow the script to start even if `datasets` is unavailable.
    try:
        from viterbo.datasets.atlas_tiny_io import (
            atlas_tiny_load_parquet,
            atlas_tiny_rows_from_hf,
        )
    except ModuleNotFoundError as e:
        print(
            "Hugging Face Datasets dependency not available (module 'datasets' missing).\n"
            "This explorer loads via atlas_tiny_io; please ensure C1 environment has 'datasets' installed.\n"
            f"Details: {e}"
        )
        return None

    print(f"Loading HF Dataset from: {parquet_path}")
    ds = atlas_tiny_load_parquet(str(parquet_path))
    print(f"Loaded HF Dataset with {ds.num_rows} rows.")
    rows = atlas_tiny_rows_from_hf(ds)
    print(f"Reconstructed {len(rows)} in-memory rows (float64 CPU tensors).")
    return rows


rows = load_rows_from_parquet(PARQUET_PATH)
if rows is None:
    # Early exit path for missing artefacts; keep the script fast and friendly.
    sys.exit(0)


# %% [markdown]
# ## Inspect: IDs, backends, and timing summary


# %%
def fmt_seconds(x: float) -> str:
    return f"{x:.6f}s"


def summarize_timings(rows: list[dict[str, Any]], keys: Iterable[str]) -> None:
    for k in keys:
        vals = [float(v) for r in rows if (v := r.get(k)) is not None]
        if not vals:
            print(f"- {k}: no values present")
            continue
        vals.sort()
        n = len(vals)
        median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        print(
            f"- {k}: count={n}, min={fmt_seconds(vals[0])}, "
            f"median={fmt_seconds(median)}, max={fmt_seconds(vals[-1])}"
        )


polytope_ids = [r["polytope_id"] for r in rows]
print(f"Row count: {len(rows)}")
print("polytope_id list:")
print("  " + ", ".join(polytope_ids))

# Backend label summaries
for label_key in ("volume_backend", "capacity_ehz_backend", "systolic_ratio_backend"):
    freq: dict[str, int] = {}
    for r in rows:
        v = r.get(label_key)
        key = "None" if v is None else str(v)
        freq[key] = freq.get(key, 0) + 1
    items = ", ".join(f"{k}:{v}" for k, v in sorted(freq.items()))
    print(f"{label_key}: {items}")

# Timing columns (min/median/max), ignoring missing values
time_keys = [k for k in rows[0].keys() if k.startswith("time_")]
print("Timing summary (min/median/max across present values):")
summarize_timings(rows, time_keys)


# %% [markdown]
# ## Batch: collate and pad a small subset


# %%
def pick_rows_for_demo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Preferred IDs; map a possible alias for the pentagon.
    preferred = [
        "unit_square",
        "regular_pentagon_rot0",
        "regular_pentagon",
        "random_hexagon_seed41",
    ]
    want: list[str] = []
    seen = set()
    have = {r["polytope_id"] for r in rows}
    for pid in preferred:
        # Accept the first match of either alias for the pentagon.
        if pid in have and pid not in seen:
            want.append(pid)
            seen.add(pid)
        if len(want) >= 3:
            break
    # Fallback: first three rows
    if len(want) < 3:
        want = list(polytope_ids[:3])
    return [r for r in rows if r["polytope_id"] in want]


demo_rows = pick_rows_for_demo(rows)
demo_ids = [r["polytope_id"] for r in demo_rows]
print(f"Selected rows for batching: {', '.join(demo_ids)}")

batch = atlas_tiny_collate_pad(demo_rows)


def tensor_shape(x: object) -> str:
    return tuple(x.shape).__repr__() if isinstance(x, torch.Tensor) else "-"


print("Batch tensors/shapes:")
for key in (
    "vertices",
    "normals",
    "offsets",
    "minimal_action_cycle",
    "vertex_mask",
    "facet_mask",
    "cycle_mask",
    "volume",
    "capacity_ehz",
    "systolic_ratio",
):
    print(f"- {key}: {tensor_shape(batch[key])}")

# Quick per-row mask stats
vm = (
    batch["vertex_mask"].sum(dim=1).tolist()
    if isinstance(batch["vertex_mask"], torch.Tensor)
    else []
)
fm = (
    batch["facet_mask"].sum(dim=1).tolist() if isinstance(batch["facet_mask"], torch.Tensor) else []
)
cm = (
    batch["cycle_mask"].sum(dim=1).tolist() if isinstance(batch["cycle_mask"], torch.Tensor) else []
)
if vm and fm:
    print("Per-row counts — vertices/facets/cycle:")
    for i, pid in enumerate(batch["polytope_id"]):
        cyc = int(cm[i]) if cm else 0
        print(f"  {pid}: V={int(vm[i])}, F={int(fm[i])}, C={cyc}")


# %% [markdown]
# ## Visualize: 2D polygons with minimal_action_cycle overlay


# %%
def plot_polygon_2d(
    vertices: torch.Tensor, *, line_style: str = "-", label: str | None = None
) -> None:
    assert vertices.ndim == 2 and vertices.size(1) == 2
    xs = vertices[:, 0].detach().cpu().numpy()
    ys = vertices[:, 1].detach().cpu().numpy()
    plt.plot(xs, ys, line_style, lw=1.5, label=label)


def plot_row_2d(r: dict[str, Any]) -> None:
    pid = r["polytope_id"]
    verts = r["vertices"]
    if not isinstance(verts, torch.Tensor) or verts.ndim != 2 or verts.size(1) != 2:
        return
    cyc = r.get("minimal_action_cycle")

    plt.figure(figsize=(4.0, 4.0))
    # Prefer the cycle (ordered boundary) if present; otherwise connect vertices in input order.
    if isinstance(cyc, torch.Tensor) and cyc.numel() > 0 and cyc.size(1) == 2:
        plot_polygon_2d(torch.vstack([cyc, cyc[:1]]), line_style="-", label="cycle")
    plot_polygon_2d(torch.vstack([verts, verts[:1]]), line_style=":", label="vertices")
    plt.scatter(verts[:, 0].cpu(), verts[:, 1].cpu(), s=20, c="k", alpha=0.7)
    plt.axis("equal")
    plt.title(f"2D: {pid}")
    plt.legend(loc="best")
    out = PUBLISH_DIR / f"{pid}_2d.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    if SHOW_INLINE:
        plt.show()
    else:
        plt.close()
    print(f"Saved 2D plot: {out}")


for r in demo_rows:
    if int(r["dimension"]) == 2:
        plot_row_2d(r)


# %% [markdown]
# ## Visualize: 4D minimal_action_cycle — Q vs P projections


# %%
def plot_cycle_qp_4d(r: dict[str, Any]) -> bool:
    cyc = r.get("minimal_action_cycle")
    if not (
        isinstance(cyc, torch.Tensor) and cyc.ndim == 2 and cyc.size(1) == 4 and cyc.size(0) > 0
    ):
        return False
    pid = r["polytope_id"]
    q = cyc[:, :2]
    p = cyc[:, 2:]
    plt.figure(figsize=(8.0, 4.0))
    plt.subplot(1, 2, 1)
    plot_polygon_2d(torch.vstack([q, q[:1]]), line_style="-", label="Q")
    plt.scatter(q[:, 0].cpu(), q[:, 1].cpu(), s=15, c="k", alpha=0.7)
    plt.axis("equal")
    plt.title(f"4D cycle — Q proj: {pid}")
    plt.subplot(1, 2, 2)
    plot_polygon_2d(torch.vstack([p, p[:1]]), line_style="-", label="P")
    plt.scatter(p[:, 0].cpu(), p[:, 1].cpu(), s=15, c="k", alpha=0.7)
    plt.axis("equal")
    plt.title(f"4D cycle — P proj: {pid}")
    out = PUBLISH_DIR / f"{pid}_4d_cycle_qp.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    if SHOW_INLINE:
        plt.show()
    else:
        plt.close()
    print(f"Saved 4D Q/P projection plot: {out}")
    return True


plotted_4d = False
for r in rows:
    if int(r["dimension"]) == 4:
        plotted_4d = plot_cycle_qp_4d(r)
        if plotted_4d:
            break
if not plotted_4d:
    print("No 4D row with minimal_action_cycle available for Q/P projection plot.")


# %% [markdown]
# ## Notes
# - If the Parquet artefact is missing, this script exits early with a helpful message.
# - Plots are saved under `artefacts/published/atlas_tiny_explorer/` and shown inline when run in a notebook.
# - All tensors remain on CPU with float64 dtype.
