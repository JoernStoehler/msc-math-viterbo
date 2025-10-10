"""Demonstration notebook for consuming the modern atlas dataset.

This script sketches how downstream experiments might load, filter, and analyse
entries from the modern atlas. Every interaction uses stub helpers from
:mod:`viterbo.modern`, so the notebook mainly records the intended flow for
future implementations.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from viterbo.modern import converters, datasets
from viterbo.modern.types import QuantityRecord

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
    schema = datasets.atlas_schema()
    print("Loaded atlas with schema:", schema)

    head_row = atlas.row(0, named=True)
    series = pl.Series("row", [head_row])
    bundle = converters.bundle_from_row(series)
    quantities = converters.row_from_bundle_and_quantities(bundle, QuantityRecord())
    print("Prepared bundle and quantities:", bundle, quantities)
except (FileNotFoundError, NotImplementedError):
    print("Atlas consumer executed placeholder workflow.")
