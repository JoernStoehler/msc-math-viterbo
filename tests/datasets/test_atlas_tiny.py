import torch

from viterbo.datasets.atlas_tiny import (
    AtlasTinyRow,
    atlas_tiny_build,
    atlas_tiny_collate_pad,
    atlas_tiny_complete_row,
    atlas_tiny_generate,
)


def test_atlas_tiny_generate_deterministic() -> None:
    rows_a = atlas_tiny_generate()
    rows_b = atlas_tiny_generate()
    assert len(rows_a) == len(rows_b) > 0
    for raw_a, raw_b in zip(rows_a, rows_b):
        assert raw_a["polytope_id"] == raw_b["polytope_id"]
        assert raw_a["generator"] == raw_b["generator"]
        assert torch.equal(raw_a["vertices"], raw_b["vertices"])
        assert torch.equal(raw_a["normals"], raw_b["normals"])
        assert torch.equal(raw_a["offsets"], raw_b["offsets"])


def test_atlas_tiny_complete_row_attaches_invariants() -> None:
    raw_row = atlas_tiny_generate()[0]
    completed: AtlasTinyRow = atlas_tiny_complete_row(raw_row)

    assert completed["volume"].ndim == 0
    assert completed["num_vertices"] == completed["vertices"].size(0)
    assert completed["num_facets"] == completed["normals"].size(0)
    assert completed["dimension"] == completed["vertices"].size(1)

    if completed["capacity_ehz"] is not None:
        assert completed["capacity_ehz"].ndim == 0
        assert completed["systolic_ratio"] is not None
        assert completed["systolic_ratio"].ndim == 0
        assert completed["minimal_action_cycle"] is not None
        assert completed["minimal_action_cycle"].dim() == 2


def test_atlas_tiny_build_returns_completed_rows() -> None:
    rows1 = atlas_tiny_build()
    rows2 = atlas_tiny_build()

    assert isinstance(rows1, list)
    assert len(rows1) == len(rows2) > 0

    for row_a, row_b in zip(rows1, rows2):
        assert row_a["polytope_id"] == row_b["polytope_id"]
        assert torch.equal(row_a["vertices"], row_b["vertices"])
        assert torch.equal(row_a["normals"], row_b["normals"])
        assert torch.equal(row_a["offsets"], row_b["offsets"])
        assert torch.equal(row_a["volume"], row_b["volume"])


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
        assert torch.all(batch["vertices"][idx, :num_vertices] == row["vertices"])
        assert torch.all(batch["normals"][idx, :num_facets] == row["normals"])
        assert torch.all(batch["offsets"][idx, :num_facets] == row["offsets"])

        assert torch.all(batch["cycle_mask"][idx, :cycle_len])
        assert not torch.any(batch["cycle_mask"][idx, cycle_len:])
        if cycle_len > 0 and cycle is not None:
            assert torch.all(batch["minimal_action_cycle"][idx, :cycle_len] == cycle)
