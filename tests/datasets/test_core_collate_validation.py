import pytest
import torch

from viterbo.datasets.core import (
    RaggedPointsDataset,
    Sample,
    _validate_ragged_batch,
    collate_list,
    collate_pad,
)


def make(points, direction) -> Sample:
    return Sample(points=torch.as_tensor(points), direction=torch.as_tensor(direction))


def test_validate_empty_batch_raises() -> None:
    with pytest.raises(ValueError):
        _validate_ragged_batch([], caller="unit")


def test_validate_non_sample_raises() -> None:
    s = make([[1.0, 0.0]], [1.0, 0.0])
    with pytest.raises(ValueError):
        _validate_ragged_batch([s, 123], caller="unit")  # type: ignore[list-item]


def test_validate_first_item_not_sample_raises() -> None:
    s = make([[1.0, 0.0]], [1.0, 0.0])
    with pytest.raises(ValueError):
        _validate_ragged_batch([123, s], caller="unit")  # type: ignore[list-item]


def test_validate_malformed_first_shapes_raises() -> None:
    # points not 2D
    bad_points = Sample(points=torch.ones((3,)), direction=torch.ones((3,)))
    with pytest.raises(ValueError):
        _validate_ragged_batch([bad_points], caller="unit")

    # direction not 1D
    bad_dir = Sample(points=torch.ones((2, 3)), direction=torch.ones((1, 3)))
    with pytest.raises(ValueError):
        _validate_ragged_batch([bad_dir], caller="unit")


def test_validate_mismatched_D_on_first_raises() -> None:
    sample = Sample(points=torch.ones((2, 3)), direction=torch.ones((2,)))
    with pytest.raises(ValueError):
        _validate_ragged_batch([sample], caller="unit")


def test_validate_mismatched_dtype_raises() -> None:
    a = Sample(
        points=torch.ones((1, 2), dtype=torch.float32),
        direction=torch.ones((2,), dtype=torch.float32),
    )
    b = Sample(
        points=torch.ones((1, 2), dtype=torch.float64),
        direction=torch.ones((2,), dtype=torch.float64),
    )
    with pytest.raises(ValueError):
        _validate_ragged_batch([a, b], caller="unit")


def test_validate_first_item_direction_dtype_mismatch_raises() -> None:
    bad = Sample(
        points=torch.ones((2, 3), dtype=torch.float32),
        direction=torch.ones((3,), dtype=torch.float64),
    )
    with pytest.raises(ValueError):
        _validate_ragged_batch([bad], caller="unit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_validate_mismatched_device_raises() -> None:
    a = Sample(points=torch.ones((1, 2), device="cpu"), direction=torch.ones((2,), device="cpu"))
    b = Sample(
        points=torch.ones((1, 2), device=torch.device("cuda")),
        direction=torch.ones((2,), device=torch.device("cuda")),
    )
    with pytest.raises(ValueError):
        _validate_ragged_batch([a, b], caller="unit")


def test_validate_malformed_later_items_and_lengths() -> None:
    # First item is well-formed; second triggers different branches.
    good = Sample(points=torch.randn(3, 4), direction=torch.randn(4))

    # Wrong points feature dimension on later item
    wrong_D = Sample(points=torch.randn(2, 5), direction=torch.randn(4))
    with pytest.raises(ValueError):
        _validate_ragged_batch([good, wrong_D], caller="unit")

    # Wrong direction length on later item
    wrong_dir = Sample(points=torch.randn(2, 4), direction=torch.randn(5))
    with pytest.raises(ValueError):
        _validate_ragged_batch([good, wrong_dir], caller="unit")

    # Wrong ranks on later items
    bad_points_rank = Sample(points=torch.randn(4), direction=torch.randn(4))
    with pytest.raises(ValueError):
        _validate_ragged_batch([good, bad_points_rank], caller="unit")

    bad_dir_rank = Sample(points=torch.randn(2, 4), direction=torch.randn(1, 4))
    with pytest.raises(ValueError):
        _validate_ragged_batch([good, bad_dir_rank], caller="unit")

    # Finally, lengths are collected as the number of points per sample
    s1 = Sample(points=torch.zeros(0, 4), direction=torch.zeros(4))
    s2 = Sample(points=torch.zeros(2, 4), direction=torch.zeros(4))
    s3 = Sample(points=torch.zeros(5, 4), direction=torch.zeros(4))
    dim, device, dtype, lengths = _validate_ragged_batch([s1, s2, s3], caller="unit")
    assert dim == 4
    assert device == s1.points.device and dtype == s1.points.dtype
    assert lengths == [0, 2, 5]


def test_collate_list_happy_path() -> None:
    s1 = make([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0])
    s2 = make([[5.0, 6.0]], [1.0, 0.0])
    out = collate_list([s1, s2])
    assert isinstance(out["points"], list) and isinstance(out["direction"], list)
    assert out["points"][0].shape == (2, 2) and out["points"][1].shape == (1, 2)
    assert torch.all(out["points"][0] == s1.points)
    assert torch.all(out["points"][1] == s2.points)
    assert torch.all(out["direction"][0] == s1.direction)
    assert torch.all(out["direction"][1] == s2.direction)


def test_collate_pad_happy_path_mask_and_values() -> None:
    a = make([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0])
    b = make([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], [1.0, 0.0])
    out = collate_pad([a, b])
    assert out["points"].shape == (2, 3, 2)
    assert out["direction"].shape == (2, 2)
    # Values and mask
    assert torch.all(out["points"][0, :2] == a.points)
    assert torch.all(out["points"][1, :3] == b.points)
    assert torch.equal(out["mask"][0], torch.tensor([True, True, False]))
    assert torch.equal(out["mask"][1], torch.tensor([True, True, True]))


def test_collate_pad_handles_zero_length_sample() -> None:
    s = Sample(points=torch.zeros((0, 3)), direction=torch.zeros((3,)))
    out = collate_pad([s])
    assert out["points"].shape == (1, 0, 3)
    assert out["mask"].shape == (1, 0)


def test_ragged_points_dataset_seed_none_and_shapes() -> None:
    ds = RaggedPointsDataset(num_samples=2, dim=3, min_points=1, max_points=2, seed=None)
    assert len(ds) == 2
    sample = ds[0]
    assert isinstance(sample, Sample)
    assert sample.points.ndim == 2 and sample.points.size(1) == 3
    assert sample.direction.shape == (3,)


pytestmark = pytest.mark.smoke
