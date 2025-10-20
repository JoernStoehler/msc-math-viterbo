import os
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from viterbo.datasets.atlas_tiny import AtlasTinyRow, atlas_tiny_build
from viterbo.datasets.atlas_tiny_io import (
    atlas_tiny_load_parquet,
    atlas_tiny_rows_from_hf,
    atlas_tiny_save_parquet,
    atlas_tiny_to_hf,
)

pytestmark = pytest.mark.smoke


def _rows_equal(a: AtlasTinyRow, b: AtlasTinyRow) -> None:
    assert a["polytope_id"] == b["polytope_id"]
    assert a["generator"] == b["generator"]
    assert a["generator_config"] == b["generator_config"]
    assert a["dimension"] == b["dimension"]
    assert a["num_vertices"] == b["num_vertices"]
    assert a["num_facets"] == b["num_facets"]
    assert torch.allclose(a["vertices"], b["vertices"], atol=0.0, rtol=0.0)
    assert torch.allclose(a["normals"], b["normals"], atol=0.0, rtol=0.0)
    assert torch.allclose(a["offsets"], b["offsets"], atol=0.0, rtol=0.0)

    if a["minimal_action_cycle"] is None:
        assert b["minimal_action_cycle"] is None
    else:
        assert b["minimal_action_cycle"] is not None
        assert torch.allclose(
            cast(torch.Tensor, a["minimal_action_cycle"]),
            cast(torch.Tensor, b["minimal_action_cycle"]),
            atol=0.0,
            rtol=0.0,
        )

    assert torch.allclose(a["volume"], b["volume"], atol=1e-12, rtol=1e-12)
    if a["capacity_ehz"] is None:
        assert b["capacity_ehz"] is None
    else:
        assert b["capacity_ehz"] is not None
        assert torch.allclose(
            cast(torch.Tensor, a["capacity_ehz"]),
            cast(torch.Tensor, b["capacity_ehz"]),
            atol=1e-12,
            rtol=1e-12,
        )
    if a["systolic_ratio"] is None:
        assert b["systolic_ratio"] is None
    else:
        assert b["systolic_ratio"] is not None
        assert torch.allclose(
            cast(torch.Tensor, a["systolic_ratio"]),
            cast(torch.Tensor, b["systolic_ratio"]),
            atol=1e-12,
            rtol=1e-12,
        )

    assert a["volume_backend"] == b["volume_backend"]
    assert a["capacity_ehz_backend"] == b["capacity_ehz_backend"]
    assert a["systolic_ratio_backend"] == b["systolic_ratio_backend"]

    # Walltimes: presence/None parity; numeric tolerance is trivial here
    assert (a["time_volume_area2d"] is None) == (b["time_volume_area2d"] is None)
    assert (a["time_volume_facets"] is None) == (b["time_volume_facets"] is None)
    assert (a["time_capacity_area2d"] is None) == (b["time_capacity_area2d"] is None)
    assert (a["time_capacity_minkowski_lp3"] is None) == (b["time_capacity_minkowski_lp3"] is None)
    assert (a["time_systolic_ratio"] is None) == (b["time_systolic_ratio"] is None)


def test_roundtrip_hf_parquet(tmp_path) -> None:
    rows = atlas_tiny_build()
    ds = atlas_tiny_to_hf(rows)

    out_dir = tmp_path / "artefacts" / "datasets" / "atlas-tiny" / "v1"
    atlas_tiny_save_parquet(ds, str(out_dir))

    # Ensure files written
    assert (out_dir / "data.parquet").is_file()
    assert (out_dir / "dataset_info.json").is_file()
    assert (out_dir / "metadata.json").is_file()
    assert (out_dir / "README.md").is_file()

    loaded = atlas_tiny_load_parquet(str(out_dir))
    rows_back = atlas_tiny_rows_from_hf(loaded)

    assert len(rows_back) == len(rows) > 0
    for a, b in zip(rows, rows_back):
        _rows_equal(a, b)


def test_schema_types_and_nullability(tmp_path) -> None:
    rows = atlas_tiny_build()
    ds = atlas_tiny_to_hf(rows)
    out_dir = tmp_path / "parquet"
    atlas_tiny_save_parquet(ds, str(out_dir))

    table = pq.read_table(os.fspath(out_dir / "data.parquet"))
    schema = table.schema

    def field(name: str) -> pa.Field:
        f = schema.field(name)
        assert f is not None
        return f

    # Identity/meta
    assert pa.types.is_string(field("polytope_id").type)
    assert pa.types.is_string(field("generator").type)
    assert pa.types.is_string(field("generator_config").type)
    assert pa.types.is_int64(field("dimension").type)
    assert pa.types.is_int64(field("num_vertices").type)
    assert pa.types.is_int64(field("num_facets").type)

    # Geometry types
    v_type = field("vertices").type
    assert pa.types.is_list(v_type)
    assert pa.types.is_list(v_type.value_type)
    assert pa.types.is_float64(v_type.value_type.value_type)

    n_type = field("normals").type
    assert pa.types.is_list(n_type)
    assert pa.types.is_list(n_type.value_type)
    assert pa.types.is_float64(n_type.value_type.value_type)

    o_type = field("offsets").type
    assert pa.types.is_list(o_type)
    assert pa.types.is_float64(o_type.value_type)

    c_type = field("minimal_action_cycle").type
    assert pa.types.is_list(c_type)
    assert pa.types.is_list(c_type.value_type)
    assert pa.types.is_float64(c_type.value_type.value_type)

    # Scalars and labels
    assert pa.types.is_float64(field("volume").type)
    assert pa.types.is_float64(field("capacity_ehz").type)
    assert pa.types.is_float64(field("systolic_ratio").type)
    assert pa.types.is_string(field("volume_backend").type)
    assert pa.types.is_string(field("capacity_ehz_backend").type)
    assert pa.types.is_string(field("systolic_ratio_backend").type)

    # Timings
    for name in [
        "time_generator",
        "time_volume_area2d",
        "time_volume_facets",
        "time_capacity_area2d",
        "time_capacity_minkowski_lp3",
        "time_systolic_ratio",
    ]:
        assert pa.types.is_float64(field(name).type)


def test_timings_presence_rules(tmp_path) -> None:
    rows = atlas_tiny_build()
    ds = atlas_tiny_to_hf(rows)
    out_dir = tmp_path / "parquet2"
    atlas_tiny_save_parquet(ds, str(out_dir))
    loaded = atlas_tiny_load_parquet(str(out_dir))

    for rec in loaded:
        dim = int(rec["dimension"])

        if dim == 2:
            assert rec["time_volume_area2d"] is not None and rec["time_volume_area2d"] >= 0.0
            assert rec["time_volume_facets"] is None
            assert rec["time_capacity_area2d"] is not None and rec["time_capacity_area2d"] >= 0.0
            assert rec["time_capacity_minkowski_lp3"] is None
            assert rec["time_systolic_ratio"] is not None and rec["time_systolic_ratio"] >= 0.0
        else:
            assert rec["time_volume_area2d"] is None
            assert rec["time_volume_facets"] is not None and rec["time_volume_facets"] >= 0.0
            if rec["capacity_ehz_backend"] == "minkowski_lp3":
                assert (
                    rec["time_capacity_minkowski_lp3"] is not None
                    and rec["time_capacity_minkowski_lp3"] >= 0.0
                )
            else:
                assert rec["time_capacity_minkowski_lp3"] is None

            if rec["systolic_ratio_backend"] == "formula":
                assert rec["time_systolic_ratio"] is not None and rec["time_systolic_ratio"] >= 0.0
            else:
                assert rec["time_systolic_ratio"] is None

        # Minimal action cycle persistence rule: empty list when missing
        if rec["capacity_ehz"] is None:
            assert rec["minimal_action_cycle"] == []
        else:
            assert (
                isinstance(rec["minimal_action_cycle"], list)
                and len(rec["minimal_action_cycle"]) >= 1
            )
