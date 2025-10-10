"""Conversion helpers between Polars rows and JAX-friendly structures."""

from __future__ import annotations

import polars as pl

from .types import PolytopeBundle, QuantityRecord


def bundle_from_row(row: pl.Series) -> PolytopeBundle:
    """Build a :class:`PolytopeBundle` from a dataframe row."""

    raise NotImplementedError


def row_from_bundle_and_quantities(
    bundle: PolytopeBundle,
    quantities: QuantityRecord,
) -> pl.Series:
    """Construct a Polars row for ``bundle`` and ``quantities``."""

    raise NotImplementedError
