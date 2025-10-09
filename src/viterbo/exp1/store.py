from __future__ import annotations

"""Imperative dataset utilities (stubs).

Keep storage logic straightforward: single Parquet table per dataset, functions
to ensure existence, append rows, scan lazily, and load subsets. Avoid extra
abstraction layers; get to math code quickly.

MVP Row Schema ("MvpRow")
-------------------------
Rows represent polytopes with the following columns. Not all fields are
applicable to every polytope; absent values should be stored as nulls.

Required
- ``polytope_id: str`` — stable identifier for joining and deduplication.
- ``generator: str`` — family/generator label (e.g., "products.km_xtk").
- ``dimension: int`` — ambient dimension (typically 4; some 2 and 6).

Representations (nullable)
- ``hrep_normals: list[list[float64]]`` — outer length = n_facets;
  inner fixed size = ``dimension`` (Arrow FixedSizeList recommended).
- ``hrep_offsets: list[float64]`` — length = n_facets.
- ``vrep_vertices: list[list[float64]]`` — outer length = n_vertices;
  inner fixed size = ``dimension`` (Arrow FixedSizeList recommended).
- ``is_lagrangian_product: bool`` — whether shape is a product of 2D factors.

Quantities (nullable per availability)
- ``volume: float64`` — natural scale volume.
- ``capacity_ehz: float64`` — EHZ capacity estimate.
- ``systolic_ratio: float64`` — scale-invariant ratio.
- ``min_action_orbit: list[int64]`` — facet indices for the minimizing word.

Notes on types
- Store numeric scalars as float64; arrays as Arrow lists with float64 elements.
- Encode coordinate dimension via inner fixed-size arrays; infer counts via
  list lengths (`arr.lengths()` in Polars) to avoid redundant columns.
"""

from typing import Any, Iterable, Mapping, Sequence, TypedDict


class MvpRow(TypedDict, total=False):
    """Typed mapping for MVP dataset rows.

    Keys are optional at the type level to allow partial updates and nullable
    fields per polytope family. Callers should include at least the required
    keys (``polytope_id``, ``generator``, ``dimension``) when appending new rows.
    """

    polytope_id: str
    generator: str
    dimension: int
    hrep_normals: list[list[float]]
    hrep_offsets: list[float]
    vrep_vertices: list[list[float]]
    is_lagrangian_product: bool
    volume: float
    capacity_ehz: float
    systolic_ratio: float
    min_action_orbit: list[int]


def ensure_dataset(path: str) -> None:
    """Create an empty dataset file at ``path`` if missing.

    Implementations may write an empty Parquet with the MVP schema or defer
    schema materialization until first append.
    """

    ...


def append_rows(path: str, rows: Iterable[MvpRow]) -> None:
    """Append row dictionaries to the dataset at ``path``.

    MVP semantics may read existing rows, concatenate, and rewrite the file.
    """

    ...


def scan_lazy(path: str) -> Any:
    """Return a lazy scan object for the dataset at ``path``.

    Intended to be a Polars ``LazyFrame`` via the `_wrapped.polars_io` module.
    The return type is ``Any`` to avoid hard dependency in the exp1 layer.
    """

    ...


def load_rows(path: str, columns: Sequence[str] | None = None) -> Any:
    """Load rows into a DataFrame-like object, optionally selecting columns.

    Should delegate to the Polars wrapper read function.
    """

    ...


def select_halfspaces_volume_capacity(
    path: str, polytope_ids: Sequence[str] | None = None
) -> Any:
    """Return halfspace data, volume, and capacity for selected polytopes.

    Args:
      path: Dataset file path.
      polytope_ids: Optional subset to select; when None, return all.

    Returns:
      A DataFrame-like object or an iterable of dicts with keys:
      ``hrep_normals``, ``hrep_offsets``, ``volume``, ``capacity_ehz``.
    """

    ...


def log_row(poly: Any, quantities: Mapping[str, Any]) -> MvpRow:
    """Build a single dataset row for ``poly`` and computed ``quantities``.

    The minimal viable row schema includes:
      - ``polytope_id: str``
      - ``generator: str``
      - ``dimension: int``
      - ``hrep_normals: list[list[float]]`` (nullable)
      - ``hrep_offsets: list[float]`` (nullable)
      - ``vrep_vertices: list[list[float]]`` (nullable)
      - ``is_lagrangian_product: bool``
      - ``volume: float``
      - ``capacity_ehz: float``
      - ``systolic_ratio: float``
      - ``min_action_orbit: list[int]`` (facet indices)

    Args:
      poly: Polytope object (H-rep, V-rep, or product form). Caller may convert
        representations before logging if needed.
      quantities: Mapping with computed scalar/list values (e.g., volume,
        capacity, sys, min_action_orbit). Missing items are treated as absent.

    Returns:
      A dictionary (``MvpRow``) ready to append to the dataset.
    """

    ...
