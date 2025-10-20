"""Hugging Face Datasets + Parquet I/O for AtlasTiny.

This module isolates all storage/backends for AtlasTiny. Numeric data is
persisted as float64; ragged geometry uses nested lists (list[list[float64]]).

Functions:
- ``atlas_tiny_to_hf``: build a ``datasets.Dataset`` from completed rows.
- ``atlas_tiny_save_parquet``: write Parquet and companion metadata files.
- ``atlas_tiny_load_parquet``: load Parquet back into a ``datasets.Dataset``.
- ``atlas_tiny_rows_from_hf``: reconstruct in-memory ``AtlasTinyRow`` objects.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, cast

import torch

import datasets as hf


def _ensure_float64_cpu_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float64, device=torch.device("cpu"))


def _tensor_to_nested_list(x: torch.Tensor) -> list[list[float]]:
    if x.ndim != 2:
        raise ValueError("expected 2D tensor for nested list conversion")
    x = _ensure_float64_cpu_tensor(x)
    return x.tolist()  # list[list[float]]


def _tensor1d_to_list(x: torch.Tensor) -> list[float]:
    if x.ndim != 1:
        raise ValueError("expected 1D tensor for list conversion")
    x = _ensure_float64_cpu_tensor(x)
    return x.tolist()


def _scalar_tensor_to_float(x: torch.Tensor) -> float:
    if x.ndim != 0:
        raise ValueError("expected scalar tensor")
    return float(x.item())


def atlas_tiny_to_hf(rows: list[dict[str, Any]]) -> hf.Dataset:
    """Construct an HF Dataset from completed AtlasTiny rows.

    - Enforces float64 CPU for numeric tensors.
    - Uses nested lists for ragged geometry (vertices, normals, offsets, cycle).
    - For ``minimal_action_cycle``: None → empty list.
    - Nullable scalars map to Arrow nulls via Python ``None``.
    """

    if not rows:
        # Create an empty dataset with declared schema when no rows are provided.
        raise ValueError("atlas_tiny_to_hf requires at least one row")

    features = hf.Features(
        {
            # Identity/meta
            "polytope_id": hf.Value("string"),
            "generator": hf.Value("string"),
            "generator_config": hf.Value("string"),
            "dimension": hf.Value("int64"),
            "num_vertices": hf.Value("int64"),
            "num_facets": hf.Value("int64"),
            # Geometry
            "vertices": hf.Sequence(hf.Sequence(hf.Value("float64"))),
            "normals": hf.Sequence(hf.Sequence(hf.Value("float64"))),
            "offsets": hf.Sequence(hf.Value("float64")),
            "minimal_action_cycle": hf.Sequence(hf.Sequence(hf.Value("float64"))),
            # Quantities
            "volume": hf.Value("float64"),
            "capacity_ehz": hf.Value("float64"),  # nullable via None
            "systolic_ratio": hf.Value("float64"),  # nullable via None
            # Backend labels
            "volume_backend": hf.Value("string"),
            "capacity_ehz_backend": hf.Value("string"),  # nullable
            "systolic_ratio_backend": hf.Value("string"),  # nullable
            # Walltimes
            "time_generator": hf.Value("float64"),
            "time_volume_area2d": hf.Value("float64"),
            "time_volume_facets": hf.Value("float64"),
            "time_capacity_area2d": hf.Value("float64"),
            "time_capacity_minkowski_lp3": hf.Value("float64"),
            "time_systolic_ratio": hf.Value("float64"),
        }
    )

    records: list[dict[str, Any]] = []
    for r in rows:
        # Basic integrity checks and coercions
        v = _ensure_float64_cpu_tensor(r["vertices"])
        n = _ensure_float64_cpu_tensor(r["normals"])
        o = _ensure_float64_cpu_tensor(r["offsets"])
        if v.ndim != 2 or n.ndim != 2 or o.ndim != 1:
            raise ValueError("invalid geometry shapes; expected (M,D),(F,D),(F,)")

        cycle = r["minimal_action_cycle"]
        cycle_list: list[list[float]] = [] if cycle is None else _tensor_to_nested_list(cycle)

        volume = _scalar_tensor_to_float(_ensure_float64_cpu_tensor(r["volume"]))
        capacity = (
            None
            if r["capacity_ehz"] is None
            else _scalar_tensor_to_float(_ensure_float64_cpu_tensor(r["capacity_ehz"]))
        )
        systolic = (
            None
            if r["systolic_ratio"] is None
            else _scalar_tensor_to_float(_ensure_float64_cpu_tensor(r["systolic_ratio"]))
        )

        rec: dict[str, Any] = {
            # Identity/meta
            "polytope_id": r["polytope_id"],
            "generator": r["generator"],
            "generator_config": r["generator_config"],
            "dimension": int(r["dimension"]),
            "num_vertices": int(r["num_vertices"]),
            "num_facets": int(r["num_facets"]),
            # Geometry
            "vertices": _tensor_to_nested_list(v),
            "normals": _tensor_to_nested_list(n),
            "offsets": _tensor1d_to_list(o),
            "minimal_action_cycle": cycle_list,
            # Quantities (nullable via None)
            "volume": volume,
            "capacity_ehz": capacity,
            "systolic_ratio": systolic,
            # Backend labels (nullable)
            "volume_backend": r["volume_backend"],
            "capacity_ehz_backend": r.get("capacity_ehz_backend"),
            "systolic_ratio_backend": r.get("systolic_ratio_backend"),
            # Walltimes (nullable)
            "time_generator": float(r["time_generator"]),
            "time_volume_area2d": r.get("time_volume_area2d"),
            "time_volume_facets": r.get("time_volume_facets"),
            "time_capacity_area2d": r.get("time_capacity_area2d"),
            "time_capacity_minkowski_lp3": r.get("time_capacity_minkowski_lp3"),
            "time_systolic_ratio": r.get("time_systolic_ratio"),
        }
        records.append(rec)

    ds = hf.Dataset.from_list(records, features=features)
    return ds


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class AtlasTinyMetadata:
    """Export metadata for AtlasTiny Parquet artefacts."""

    commit: str
    build_time_utc: str
    schema_version: str
    row_count: int
    seeds_or_config_summary: str


def atlas_tiny_save_parquet(ds: hf.Dataset, out_dir: str) -> None:
    """Save dataset as Parquet with companion metadata files.

    Files written under ``out_dir``:
    - ``data.parquet``: all rows/columns in a single Parquet file.
    - ``dataset_info.json``: minimal HF-like info (features + num_rows).
    - ``metadata.json``: commit hash, build time, schema version, row count, summary.
    - ``README.md``: schema summary and reproducibility notes.
    """

    os.makedirs(out_dir, exist_ok=True)

    parquet_path = os.path.join(out_dir, "data.parquet")
    # pyright: ignore[reportAttributeAccessIssue]
    cast(Any, ds).to_parquet(parquet_path)

    # dataset_info.json — keep minimal but useful
    dataset_info = {
        "description": "AtlasTiny v1 dataset (exported to Parquet)",
        # Keep a readable string summary of features; exact schema is in Parquet.
        "features": str(ds.features),
        "splits": {"train": {"num_bytes": None, "num_examples": ds.num_rows}},
        "download_checksums": {},
        "download_size": None,
        "dataset_size": None,
        "config_name": "atlas-tiny-v1",
        "version": "1.0.0",
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    # metadata.json — project-specific
    summary = "AtlasTiny roster with deterministic generators; see generator_config"
    meta = AtlasTinyMetadata(
        commit=_git_commit_hash(),
        build_time_utc=datetime.now(tz=UTC).isoformat(),
        schema_version="v1",
        row_count=ds.num_rows,
        seeds_or_config_summary=summary,
    )
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    # README.md — concise schema summary
    readme_lines = [
        "# AtlasTiny v1 (Parquet Export)",
        "",
        "Backends: Hugging Face Datasets (in-memory) + PyArrow Parquet.",
        "All numeric columns are float64; geometry is ragged via nested lists.",
        "",
        "## Columns",
        "- polytope_id (string)",
        "- generator (string)",
        "- generator_config (string; JSON)",
        "- dimension, num_vertices, num_facets (int64)",
        "- vertices: list[list[float64]]",
        "- normals: list[list[float64]]",
        "- offsets: list[float64]",
        "- minimal_action_cycle: list[list[float64]] (empty when missing)",
        "- volume: float64",
        "- capacity_ehz, systolic_ratio: float64 (nullable)",
        "- volume_backend: string",
        "- capacity_ehz_backend, systolic_ratio_backend: string (nullable)",
        "- time_*: float64 (nullable for non-executed backends)",
        "",
        "## Reproducibility",
        "- See metadata.json for commit + build timestamp.",
        "- Generator parameters encoded in generator_config (JSON).",
        "- Persisted numeric dtype: float64 CPU.",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")


def atlas_tiny_load_parquet(path: str) -> hf.Dataset:
    """Load a Parquet file exported by ``atlas_tiny_save_parquet`` into HF Dataset."""
    # Accept either directory or direct parquet file path
    parquet_path = path
    if os.path.isdir(path):
        parquet_path = os.path.join(path, "data.parquet")
    # pyright: ignore[reportAttributeAccessIssue]
    return cast(Any, hf.Dataset).from_parquet(parquet_path)


def atlas_tiny_rows_from_hf(ds: hf.Dataset) -> list[dict[str, Any]]:
    """Reconstruct in-memory ``AtlasTinyRow`` objects (float64 CPU tensors).

    - Nested lists → torch.float64 CPU tensors.
    - Empty ``minimal_action_cycle`` lists → None.
    - Nullable scalars (capacity/systolic, time_*) propagate None.
    """

    rows: list[dict[str, Any]] = []
    for rec in cast(Any, ds):
        # Geometry
        v = torch.tensor(rec["vertices"], dtype=torch.float64)
        n = torch.tensor(rec["normals"], dtype=torch.float64)
        o = torch.tensor(rec["offsets"], dtype=torch.float64)
        cyc_list = rec["minimal_action_cycle"]
        cycle = (
            None
            if (cyc_list is None or len(cyc_list) == 0)
            else torch.tensor(cyc_list, dtype=torch.float64)
        )

        # Scalars
        volume = torch.tensor(rec["volume"], dtype=torch.float64)
        cap = rec.get("capacity_ehz")
        capacity = None if cap is None else torch.tensor(cap, dtype=torch.float64)
        sys = rec.get("systolic_ratio")
        systolic = None if sys is None else torch.tensor(sys, dtype=torch.float64)

        row: dict[str, Any] = {
            # Identity/meta
            "polytope_id": rec["polytope_id"],
            "generator": rec["generator"],
            "generator_config": rec["generator_config"],
            "dimension": int(rec["dimension"]),
            "num_vertices": int(rec["num_vertices"]),
            "num_facets": int(rec["num_facets"]),
            # Geometry
            "vertices": v,
            "normals": n,
            "offsets": o,
            "minimal_action_cycle": cycle,
            # Quantities
            "volume": volume,
            "capacity_ehz": capacity,
            "systolic_ratio": systolic,
            # Backend labels
            "volume_backend": rec["volume_backend"],
            "capacity_ehz_backend": rec.get("capacity_ehz_backend"),
            "systolic_ratio_backend": rec.get("systolic_ratio_backend"),
            # Walltimes
            "time_generator": rec["time_generator"],
            "time_volume_area2d": rec.get("time_volume_area2d"),
            "time_volume_facets": rec.get("time_volume_facets"),
            "time_capacity_area2d": rec.get("time_capacity_area2d"),
            "time_capacity_minkowski_lp3": rec.get("time_capacity_minkowski_lp3"),
            "time_systolic_ratio": rec.get("time_systolic_ratio"),
        }
        rows.append(row)

    return rows
