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
# # AtlasTiny v1 — Basic Analysis (Health Check & Timing Tables)
#
# This notebook loads the AtlasTiny v1 dataset from Parquet (via Hugging Face
# Datasets helpers), reconstructs in‑memory rows, and prints concise, labeled
# tables for:
# - Roster overview and backend labels
# - Per‑row values (volume, capacity, systolic ratio) and timings
# - Aggregated timing statistics per `time_*` column
# - Invariants checks (e.g., 2D: capacity == area)
#
# Visualization is intentionally avoided; tables and printed floats are used to
# make algorithm comparisons and inconsistencies obvious at a glance.

# %%
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

# Ensure project `src/` is on the path when run as a script.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # When executed as a notebook (__file__ undefined), use the execution CWD
    # which render_notebooks.py sets to this file's parent.
    PROJECT_ROOT = Path.cwd().resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# %% [markdown]
# ## Load AtlasTiny from Parquet

# %%
PARQUET_PATH = PROJECT_ROOT / "artefacts" / "datasets" / "atlas-tiny" / "v1" / "data.parquet"


def load_rows_from_parquet(path: Path) -> list[dict[str, Any]] | None:
    """Load Parquet via atlas_tiny_io and reconstruct in‑memory rows.

    Returns None if the artefact is missing or the `datasets` dependency is not installed.
    """

    parquet_path = path / "data.parquet" if path.is_dir() else path
    if not parquet_path.exists():
        print(
            f"Parquet not found at '{parquet_path}'.\n"
            "Run: uv run python notebooks/atlas_tiny_build.py\n"
            "to create the AtlasTiny v1 artefact, then re‑run this analysis."
        )
        return None

    try:
        from viterbo.datasets.atlas_tiny_io import (
            atlas_tiny_load_parquet,
            atlas_tiny_rows_from_hf,
        )
    except ModuleNotFoundError as e:
        print(
            "Hugging Face Datasets dependency not available (module 'datasets' missing).\n"
            "Install data extras and retry: uv sync --extra data\n"
            f"Details: {e}"
        )
        return None

    ds = atlas_tiny_load_parquet(os.fspath(parquet_path))
    rows = atlas_tiny_rows_from_hf(ds)
    return rows


rows = load_rows_from_parquet(PARQUET_PATH)
if rows is None:
    sys.exit(0)


# %% [markdown]
# ## Utilities: formatting and simple table printer


# %%
def fmtf(x: float | int | None, *, digits: int = 6) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{digits}f}"


def print_table(headers: list[str], rows_: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows_:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    def line(vals: list[str]) -> str:
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)
    print(line(headers))
    print(sep)
    for r in rows_:
        print(line(r))


def median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(max(int(math.ceil(q * (len(s) - 1))), 0), len(s) - 1)
    return s[idx]


# %% [markdown]
# ## Roster overview and backend labels

# %%
polytope_ids = [r["polytope_id"] for r in rows]
dims = [int(r["dimension"]) for r in rows]
print(f"Row count: {len(rows)}")
print("polytope_id and dimension:")
print_table(
    [
        "polytope_id",
        "dim",
    ],
    [[pid, str(d)] for pid, d in zip(polytope_ids, dims)],
)


def print_label_counts(key: str) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        v = r.get(key)
        k = "None" if v is None else str(v)
        counts[k] = counts.get(k, 0) + 1
    data = [[k, str(v)] for k, v in sorted(counts.items())]
    print(f"\n{key} counts:")
    print_table([key, "count"], data)


for label_key in ("volume_backend", "capacity_ehz_backend", "systolic_ratio_backend"):
    print_label_counts(label_key)


# %% [markdown]
# ## Per‑row values and timings


# %%
def pick_time(r: dict[str, Any], keys: list[str]) -> float | None:
    for k in keys:
        if r.get(k) is not None:
            return float(r[k])
    return None


per_row_headers = [
    "polytope_id",
    "dim",
    "vol_backend",
    "cap_backend",
    "volume",
    "capacity",
    "systolic",
    "t_volume",
    "t_capacity",
    "t_systolic",
]
per_row_data: list[list[str]] = []
for r in rows:
    t_vol = pick_time(r, ["time_volume_area2d", "time_volume_facets"])  # dim-specific
    t_cap = pick_time(r, ["time_capacity_area2d", "time_capacity_minkowski_lp3"])  # nullable
    per_row_data.append(
        [
            r["polytope_id"],
            str(int(r["dimension"])),
            str(r.get("volume_backend")),
            str(r.get("capacity_ehz_backend")),
            fmtf(float(r["volume"]))
            if isinstance(r["volume"], torch.Tensor)
            else fmtf(r["volume"]),
            fmtf(
                None
                if r.get("capacity_ehz") is None
                else (
                    float(r["capacity_ehz"])
                    if not isinstance(r["capacity_ehz"], torch.Tensor)
                    else float(r["capacity_ehz"].item())
                )
            ),
            fmtf(
                None
                if r.get("systolic_ratio") is None
                else (
                    float(r["systolic_ratio"])
                    if not isinstance(r["systolic_ratio"], torch.Tensor)
                    else float(r["systolic_ratio"].item())
                )
            ),
            fmtf(t_vol),
            fmtf(t_cap),
            fmtf(None if r.get("time_systolic_ratio") is None else float(r["time_systolic_ratio"])),
        ]
    )

print("\nPer‑row values and timings:")
print_table(per_row_headers, per_row_data)


# %% [markdown]
# ## Aggregated timing statistics per `time_*`

# %%
time_keys = sorted(k for k in rows[0].keys() if k.startswith("time_"))


def aggregate_timings(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {
            "count": 0.0,
            "min": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(len(vals)),
        "min": min(vals),
        "median": median(vals),
        "mean": sum(vals) / len(vals),
        "p90": quantile(vals, 0.9),
        "max": max(vals),
    }


agg_headers = ["time_column", "count", "min", "median", "mean", "p90", "max"]
agg_rows: list[list[str]] = []
for key in time_keys:
    stats = aggregate_timings(rows, key)
    agg_rows.append(
        [
            key,
            fmtf(stats["count"], digits=0),
            fmtf(stats["min"]),
            fmtf(stats["median"]),
            fmtf(stats["mean"]),
            fmtf(stats["p90"]),
            fmtf(stats["max"]),
        ]
    )

print("\nTiming aggregates (seconds) across present values:")
print_table(agg_headers, agg_rows)


# %% [markdown]
# ## Invariants checks

# %%
# 2D capacity equals area (within tolerance)
two_d = [r for r in rows if int(r["dimension"]) == 2]
if two_d:
    tol = 1e-9
    errs: list[tuple[str, float, float, float]] = []  # (id, vol, cap, abs_err)
    for r in two_d:
        vol = (
            float(r["volume"])
            if not isinstance(r["volume"], torch.Tensor)
            else float(r["volume"].item())
        )
        cap = r.get("capacity_ehz")
        cap_val = (
            None
            if cap is None
            else (float(cap) if not isinstance(cap, torch.Tensor) else float(cap.item()))
        )
        if cap_val is None:
            errs.append((r["polytope_id"], vol, float("nan"), float("nan")))
            continue
        err = abs(vol - cap_val)
        errs.append((r["polytope_id"], vol, cap_val, err))
    print("\n2D: capacity vs area (expect equality)")
    print_table(
        ["polytope_id", "volume", "capacity", "abs_error"],
        [[pid, fmtf(vol), fmtf(cap), fmtf(err)] for pid, vol, cap, err in errs],
    )
    max_err = max((e for *_, e in errs if not math.isnan(e)), default=0.0)
    print(f"Max abs error (2D): {fmtf(max_err)} (tol={tol})")


# Systolic ratio consistency: capacity^n / (n! * volume) == systolic_ratio (when present)
with_cap = [r for r in rows if r.get("capacity_ehz") is not None]
if with_cap:
    sr_rows: list[list[str]] = []
    max_sr_err = 0.0
    for r in with_cap:
        dim = int(r["dimension"])
        n = dim // 2
        vol = (
            float(r["volume"])
            if not isinstance(r["volume"], torch.Tensor)
            else float(r["volume"].item())
        )
        cap = r["capacity_ehz"]
        cap_val = float(cap) if not isinstance(cap, torch.Tensor) else float(cap.item())
        sr = r.get("systolic_ratio")
        sr_val = (
            None
            if sr is None
            else (float(sr) if not isinstance(sr, torch.Tensor) else float(sr.item()))
        )
        # Literature-normalized systolic ratio
        expected = (cap_val**n) / (math.factorial(n) * vol)
        err = float("nan") if sr_val is None else abs(expected - sr_val)
        if not math.isnan(err):
            max_sr_err = max(max_sr_err, err)
        sr_rows.append([r["polytope_id"], str(dim), fmtf(expected), fmtf(sr_val), fmtf(err)])
    print("\nSystolic ratio check (expected = capacity^n / (n! * volume))")
    print_table(["polytope_id", "dim", "expected", "reported", "abs_error"], sr_rows)
    print(f"Max abs error (systolic ratio): {fmtf(max_sr_err)}")


# %% [markdown]
# ## Notes
# - All tables use fixed-width columns and labeled headers for advisor-friendly review.
# - For further comparisons, consider exporting CSV summaries from these tables if needed.
