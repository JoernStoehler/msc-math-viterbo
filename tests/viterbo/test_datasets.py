"""Atlas schema contract for modern dataset adapters."""

from __future__ import annotations

import polars as pl
import pytest

from viterbo import atlas


@pytest.mark.goal_code
@pytest.mark.smoke
def test_atlas_pl_schema_has_expected_columns() -> None:
    """atlas_pl_schema returns a Polars Schema including normals/offsets/vertices."""
    dim = 3
    schema = atlas.atlas_pl_schema(dim)
    assert isinstance(schema, pl.Schema)
    required = {"polytope_id", "dimension", "num_facets", "num_vertices", "normals", "offsets", "vertices"}
    for col in required:
        assert col in schema
