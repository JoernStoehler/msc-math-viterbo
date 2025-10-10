"""Ensure converter helpers remain stubbed."""

from __future__ import annotations

import polars as pl
import pytest

from viterbo.modern import converters
from viterbo.modern.types import PolytopeBundle, QuantityRecord


@pytest.mark.goal_code
@pytest.mark.smoke
def test_converter_stubs_raise_not_implemented() -> None:
    """Row conversion helpers should raise NotImplementedError for now."""

    row = pl.Series("dummy", [1])
    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    quantities = QuantityRecord()

    with pytest.raises(NotImplementedError):
        converters.bundle_from_row(row)
    with pytest.raises(NotImplementedError):
        converters.row_from_bundle_and_quantities(bundle, quantities)
