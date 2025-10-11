"""Demonstration notebook for consuming the modern atlas dataset.

This script sketches how downstream experiments might load, filter, and analyse
entries from the modern atlas. Every interaction uses stub helpers from
:mod:`viterbo.modern`, so the notebook mainly records the intended flow for
future implementations.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from viterbo.modern import atlas, capacity

ATLAS_PATH = Path("artefacts/modern_atlas.parquet")


def _load_atlas() -> pl.DataFrame | None:
    """Load the atlas snapshot if present."""
    if not ATLAS_PATH.exists():
        return None
    return pl.read_parquet(ATLAS_PATH)


try:
    atlas = _load_atlas()
    if atlas is None:
        raise FileNotFoundError("atlas snapshot missing")
    schema = atlas.atlas_pl_schema(dimension=3)  # example dimension
    print("Loaded atlas with schema:", schema)
    print("Modern capacity solvers:", capacity.available_solvers())

    head_row = atlas.row(0, named=True)
    series = pl.Series("row", [head_row])
    # TODO: Use atlas.as_polytope/as_cycle once rows are materialized
    print("Loaded schema:", schema)
except (FileNotFoundError, NotImplementedError):
    print("Atlas consumer executed placeholder workflow.")
