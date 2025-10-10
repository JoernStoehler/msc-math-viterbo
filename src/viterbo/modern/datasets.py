"""Dataset helpers for the modern API."""

from __future__ import annotations

import polars as pl

from .types import GeneratorMetadata, PolytopeBundle, QuantityRecord


def atlas_schema() -> pl.Schema:
    """Return the canonical schema for the modern atlas dataset."""

    raise NotImplementedError


def records_to_dataframe(
    records: list[tuple[PolytopeBundle, GeneratorMetadata, QuantityRecord]],
) -> pl.DataFrame:
    """Convert structured records into a Polars dataframe."""

    raise NotImplementedError


def merge_results(
    existing: pl.DataFrame,
    new: pl.DataFrame,
    *,
    conflict_policy: str,
) -> pl.DataFrame:
    """Merge two atlas dataframes according to ``conflict_policy``."""

    raise NotImplementedError
