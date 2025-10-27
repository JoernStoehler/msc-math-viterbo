import pytest
import torch

from viterbo.datasets.atlas_tiny import (
    atlas_tiny_collate_pad,
    atlas_tiny_complete_row,
    atlas_tiny_generate,
)

pytestmark = pytest.mark.smoke


def _row_by_id(polytope_id: str):
    specs = atlas_tiny_generate()
    spec = next(s for s in specs if s["polytope_id"] == polytope_id)
    return atlas_tiny_complete_row(spec)


def test_mixed_2d_4d_padding_masks_and_scalars_nan() -> None:
    # Choose a 2D with capacity present and a 4D with capacity absent
    row2d = _row_by_id("unit_square")
    row4d = _row_by_id("orthogonal_simplex_4d")

    assert row2d["dimension"] == 2
    assert row4d["dimension"] == 4

    batch = atlas_tiny_collate_pad([row2d, row4d])

    # Common feature dimension should be 4
    assert batch["vertices"].shape[0] == 2
    assert batch["vertices"].shape[2] == 4
    assert batch["normals"].shape[2] == 4

    # Row 0 (2D): data copied in first 2 dims; last 2 dims zero
    nv2, d2 = row2d["vertices"].size(0), row2d["vertices"].size(1)
    nf2 = row2d["normals"].size(0)
    assert d2 == 2
    assert torch.all(batch["vertices"][0, :nv2, :d2] == row2d["vertices"])
    assert torch.all(batch["normals"][0, :nf2, :d2] == row2d["normals"])
    assert torch.all(batch["vertices"][0, :nv2, d2:] == 0)
    assert torch.all(batch["normals"][0, :nf2, d2:] == 0)
    assert torch.all(batch["offsets"][0, :nf2] == row2d["offsets"])

    # Row 1 (4D): data unchanged in all 4 dims
    nv4, d4 = row4d["vertices"].size(0), row4d["vertices"].size(1)
    nf4 = row4d["normals"].size(0)
    assert d4 == 4
    assert torch.all(batch["vertices"][1, :nv4, :d4] == row4d["vertices"])
    assert torch.all(batch["normals"][1, :nf4, :d4] == row4d["normals"])
    assert torch.all(batch["offsets"][1, :nf4] == row4d["offsets"])

    # Masks reflect true lengths
    assert torch.all(batch["vertex_mask"][0, :nv2]) and not torch.any(batch["vertex_mask"][0, nv2:])
    assert torch.all(batch["facet_mask"][0, :nf2]) and not torch.any(batch["facet_mask"][0, nf2:])
    assert torch.all(batch["vertex_mask"][1, :nv4]) and not torch.any(batch["vertex_mask"][1, nv4:])
    assert torch.all(batch["facet_mask"][1, :nf4]) and not torch.any(batch["facet_mask"][1, nf4:])

    # Cycle masks: present for 2D, absent for non-product 4D
    cyc2 = row2d["minimal_action_cycle"]
    cyc2_len = 0 if cyc2 is None else int(cyc2.size(0))
    assert torch.all(batch["cycle_mask"][0, :cyc2_len])
    assert not torch.any(batch["cycle_mask"][0, cyc2_len:])
    # 4D simplex has no cycle/capacity in this dataset
    assert not torch.any(batch["cycle_mask"][1])

    # Scalars: capacity/systolic are numbers for 2D and NaN for 4D simplex
    assert batch["volume"].shape == (2,)
    assert torch.isfinite(batch["capacity_ehz"][0])
    assert torch.isfinite(batch["systolic_ratio"][0])
    assert torch.isnan(batch["capacity_ehz"][1])
    assert torch.isnan(batch["systolic_ratio"][1])


@pytest.mark.parametrize("bad_target", [0, -1, 3.14, "4"])
def test_target_dim_errors_non_int_and_small(bad_target) -> None:
    # Keep the batch minimal but heterogeneous in dimension (2 and 4)
    row2d = _row_by_id("unit_square")
    row4d = _row_by_id("orthogonal_simplex_4d")
    rows = [row2d, row4d]

    # All listed bad_target values should raise a ValueError
    with pytest.raises(ValueError):
        atlas_tiny_collate_pad(rows, target_dim=bad_target)  # type: ignore[arg-type]

    # Also ensure "smaller than max" case explicitly (max=4 here)
    with pytest.raises(ValueError):
        atlas_tiny_collate_pad(rows, target_dim=3)
