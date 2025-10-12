"""Atlas schema contract for modern dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, Features

from viterbo.datasets import atlas


@pytest.mark.goal_code
@pytest.mark.smoke
def test_atlas_features_has_expected_columns() -> None:
    """atlas_features exposes HF dataset fields for the atlas schema."""

    features = atlas.atlas_features()
    assert isinstance(features, Features)
    required = {
        "polytope_id",
        "dimension",
        "num_facets",
        "num_vertices",
        "normals",
        "offsets",
        "vertices",
        "minimum_action_cycle",
    }
    assert required.issubset(features.keys())


def _make_row(volume: float, suffix: str = "") -> dict[str, Any]:
    return {
        "polytope_id": f"polytope{suffix}",
        "notes": "stub",
        "distribution_name": "synthetic",
        "dimension": 2,
        "num_facets": 3,
        "num_vertices": 3,
        "normals": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        "offsets": [1.0, 1.0, 1.0],
        "vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "volume": volume,
        "ehz_capacity": 1.0,
        "systolic_ratio": 1.0,
        "minimum_action_cycle": [[0.0, 0.0]],
    }


@pytest.mark.goal_code
@pytest.mark.smoke
def test_hf_dataset_build_append_map_roundtrip(tmp_path: Path) -> None:
    """HF-backed atlas helpers build, append, map, and persist datasets."""

    base = atlas.build_dataset([_make_row(1.0, suffix="-a")])
    assert isinstance(base, Dataset)
    assert base.num_rows == 1

    appended = atlas.append_rows(base, [_make_row(2.0, suffix="-b")])
    assert appended.num_rows == 2

    mapped = atlas.map_quantities(appended, lambda row: {"volume": row["volume"] * 2})
    assert mapped.to_pandas()["volume"].tolist() == [2.0, 4.0]

    save_path = tmp_path / "atlas"
    atlas.save_dataset(mapped, save_path.as_posix())
    reloaded = atlas.load_dataset(save_path.as_posix())
    assert reloaded.num_rows == mapped.num_rows
    assert reloaded.features == mapped.features
