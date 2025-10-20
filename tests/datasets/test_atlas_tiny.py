import torch

from viterbo.datasets.atlas_tiny import (
    AtlasTinyRow,
    atlas_tiny_build,
    atlas_tiny_collate_pad,
    atlas_tiny_complete_row,
    atlas_tiny_generate,
)


def test_atlas_tiny_generate_deterministic() -> None:
    specs_a = atlas_tiny_generate()
    specs_b = atlas_tiny_generate()
    assert len(specs_a) == len(specs_b) > 0
    for spec_a, spec_b in zip(specs_a, specs_b):
        assert spec_a["polytope_id"] == spec_b["polytope_id"]
        assert spec_a["generator"] == spec_b["generator"]
        assert spec_a["generator_config"] == spec_b["generator_config"]


def test_atlas_tiny_complete_row_attaches_invariants() -> None:
    spec = atlas_tiny_generate()[0]
    completed: AtlasTinyRow = atlas_tiny_complete_row(spec)

    assert completed["volume"].ndim == 0
    assert completed["num_vertices"] == completed["vertices"].size(0)
    assert completed["num_facets"] == completed["normals"].size(0)
    assert completed["dimension"] == completed["vertices"].size(1)

    # Backend label always set for volume
    if completed["dimension"] == 2:
        assert completed["volume_backend"] == "area2d"
        assert (
            isinstance(completed["time_volume_area2d"], float)
            and completed["time_volume_area2d"] >= 0.0
        )
        assert completed["time_volume_facets"] is None
    else:
        assert completed["volume_backend"] == "facets"
        assert (
            isinstance(completed["time_volume_facets"], float)
            and completed["time_volume_facets"] >= 0.0
        )
        assert completed["time_volume_area2d"] is None

    # Capacity/systolic nullable but consistent
    if completed["capacity_ehz"] is not None:
        assert completed["capacity_ehz"].ndim == 0
        assert completed["systolic_ratio"] is not None
        assert completed["systolic_ratio"].ndim == 0
        assert completed["minimal_action_cycle"] is not None
        assert completed["minimal_action_cycle"].dim() == 2
        assert completed["systolic_ratio_backend"] == "formula"
        assert (
            isinstance(completed["time_systolic_ratio"], float)
            and completed["time_systolic_ratio"] >= 0.0
        )


def test_atlas_tiny_build_returns_completed_rows() -> None:
    rows1 = atlas_tiny_build()
    rows2 = atlas_tiny_build()

    assert isinstance(rows1, list)
    assert len(rows1) == len(rows2) > 0

    for row_a, row_b in zip(rows1, rows2):
        assert row_a["polytope_id"] == row_b["polytope_id"]
        assert row_a["generator"] == row_b["generator"]
        assert row_a["generator_config"] == row_b["generator_config"]
        assert torch.equal(row_a["vertices"], row_b["vertices"])
        assert torch.equal(row_a["normals"], row_b["normals"])
        assert torch.equal(row_a["offsets"], row_b["offsets"])
        assert torch.equal(row_a["volume"], row_b["volume"])

    # Schema presence and backends
    product_ids = {"pentagon_product_counterexample", "noisy_pentagon_product"}
    for row in rows1:
        d = row["dimension"]
        assert row["volume"].ndim == 0
        if d == 2:
            assert row["volume_backend"] == "area2d"
            assert row["capacity_ehz_backend"] == "area2d"
            assert row["capacity_ehz"] is not None
            assert torch.isclose(row["capacity_ehz"], row["volume"], atol=1e-10, rtol=1e-10)
            assert (
                isinstance(row["time_capacity_area2d"], float)
                and row["time_capacity_area2d"] >= 0.0
            )
            assert row["time_capacity_minkowski_lp3"] is None
        else:
            assert row["volume_backend"] == "facets"
            if row["polytope_id"] in product_ids:
                assert row["capacity_ehz_backend"] == "minkowski_lp3"
                assert row["capacity_ehz"] is not None
                assert (
                    row["minimal_action_cycle"] is not None
                    and row["minimal_action_cycle"].size(1) == 4
                )
                assert (
                    isinstance(row["time_capacity_minkowski_lp3"], float)
                    and row["time_capacity_minkowski_lp3"] >= 0.0
                )
            else:
                assert row["capacity_ehz"] is None
                assert row["capacity_ehz_backend"] is None
                assert row["minimal_action_cycle"] is None


def test_atlas_tiny_collate_pad_shapes() -> None:
    rows = atlas_tiny_build()
    batch = atlas_tiny_collate_pad(rows)

    assert batch["vertices"].ndim == 3
    assert batch["normals"].ndim == 3
    assert batch["offsets"].ndim == 2
    assert batch["vertex_mask"].dtype == torch.bool
    assert len(batch["polytope_id"]) == len(rows)

    for idx, row in enumerate(rows):
        num_vertices = row["vertices"].size(0)
        num_facets = row["normals"].size(0)
        cycle = row["minimal_action_cycle"]
        cycle_len = 0 if cycle is None else cycle.size(0)

        assert torch.all(batch["vertex_mask"][idx, :num_vertices])
        assert not torch.any(batch["vertex_mask"][idx, num_vertices:])
        assert torch.all(batch["facet_mask"][idx, :num_facets])
        assert not torch.any(batch["facet_mask"][idx, num_facets:])
        d_v = row["vertices"].size(1)
        d_n = row["normals"].size(1)
        # Values match within the row's feature dimension; extra padded dims are zero.
        assert torch.all(batch["vertices"][idx, :num_vertices, :d_v] == row["vertices"])
        assert torch.all(batch["normals"][idx, :num_facets, :d_n] == row["normals"])
        if batch["vertices"].size(2) > d_v:
            assert torch.all(batch["vertices"][idx, :num_vertices, d_v:] == 0)
        if batch["normals"].size(2) > d_n:
            assert torch.all(batch["normals"][idx, :num_facets, d_n:] == 0)
        assert torch.all(batch["offsets"][idx, :num_facets] == row["offsets"])

        assert torch.all(batch["cycle_mask"][idx, :cycle_len])
        assert not torch.any(batch["cycle_mask"][idx, cycle_len:])
        if cycle_len > 0 and cycle is not None:
            d_c = cycle.size(1)
            assert torch.all(batch["minimal_action_cycle"][idx, :cycle_len, :d_c] == cycle)
            if batch["minimal_action_cycle"].size(2) > d_c:
                assert torch.all(batch["minimal_action_cycle"][idx, :cycle_len, d_c:] == 0)

    # Scalar quantities present (capacity/systolic NaN where absent)
    assert batch["volume"].shape == (len(rows),)
    assert batch["capacity_ehz"].shape == (len(rows),)
    assert batch["systolic_ratio"].shape == (len(rows),)
