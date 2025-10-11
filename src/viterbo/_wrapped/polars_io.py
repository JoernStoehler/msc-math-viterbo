from __future__ import annotations

"""Polars adapters for Parquet IO and JAX materialization (stubs).

This module is the interop boundary for columnar IO. Keep library code JAX-first,
and route Polars-specific operations through this wrapper to allow future swaps.
"""

from typing import Any, Iterable, Mapping, Sequence

import polars as pl  # type: ignore[import-not-found]


def rows_to_polars(rows: Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Convert an iterable of row dicts to a Polars ``DataFrame``.

    Args:
      rows: Iterable of dict-like rows conforming to the MVP schema.

    Returns:
      A Polars ``DataFrame`` with best-effort column types.
    """
    # Minimal implementation: rely on Polars to infer dtypes from row dicts.
    return pl.DataFrame(rows)


def read_parquet(path: str, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read a Parquet dataset into a Polars ``DataFrame``.

    Args:
      path: Parquet file path.
      columns: Optional subset of columns to read.
    """
    if columns is None:
        return pl.read_parquet(path)
    return pl.read_parquet(path, columns=list(columns))


def write_parquet(df: "pl.DataFrame", path: str) -> None:
    """Write a Polars ``DataFrame`` to a Parquet file at ``path``.

    Overwrite semantics are acceptable in the current MVP.
    """
    df.write_parquet(path)


def scan_parquet(path: str) -> pl.LazyFrame:
    """Return a Polars ``LazyFrame`` scanning the Parquet dataset at ``path``.

    Useful for predicates and column projection without loading full data.
    """
    return pl.scan_parquet(path)


def materialize_to_jnp(lf: "pl.LazyFrame", columns: Sequence[str]) -> tuple[Any, ...]:
    """Materialize selected columns from a lazy frame into JAX arrays.

    Args:
      lf: Polars LazyFrame.
      columns: Names of columns to materialize in order.

    Returns:
      Tuple of JAX arrays (float64 by convention for numeric columns).
    """
    import jax.numpy as jnp
    df = lf.select(list(columns)).collect()
    out: list[Any] = []
    for name in columns:
        col = df.get_column(name)
        try:
            out.append(jnp.asarray(col.to_numpy(), dtype=jnp.float64))
        except (TypeError, ValueError, AttributeError):
            out.append(col.to_list())
    return tuple(out)
