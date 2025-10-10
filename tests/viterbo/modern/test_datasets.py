"""Ensure dataset helpers remain stubbed."""

from __future__ import annotations

import polars as pl
import pytest

from viterbo.modern import datasets
from viterbo.modern.types import GeneratorMetadata, PolytopeBundle, QuantityRecord


@pytest.mark.goal_code
@pytest.mark.smoke
def test_dataset_stubs_raise_not_implemented() -> None:
    """Dataset conversion helpers should raise NotImplementedError."""

    bundle = PolytopeBundle(halfspaces=None, vertices=None)
    metadata = GeneratorMetadata(identifier="dummy", parameters={})
    quantities = QuantityRecord()
    frame = pl.DataFrame({"dummy": [1]})

    with pytest.raises(NotImplementedError):
        datasets.atlas_schema()
    with pytest.raises(NotImplementedError):
        datasets.records_to_dataframe([(bundle, metadata, quantities)])
    with pytest.raises(NotImplementedError):
        datasets.merge_results(frame, frame, conflict_policy="overwrite")
