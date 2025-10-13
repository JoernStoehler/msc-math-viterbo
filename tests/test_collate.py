import pytest
import torch

from viterbo.datasets.core import Sample, collate_list, collate_pad


def _make_sample(points: torch.Tensor, direction: torch.Tensor) -> Sample:
    return Sample(points=points, direction=direction)


def test_collate_list_single_sample_roundtrip():
    sample = _make_sample(torch.ones((2, 3)), torch.zeros(3))
    result = collate_list([sample])
    assert result["points"] == [sample.points]
    assert result["direction"] == [sample.direction]


def test_collate_list_empty_batch_raises():
    with pytest.raises(ValueError):
        collate_list([])


def test_collate_list_dtype_mismatch_raises():
    sample32 = _make_sample(
        torch.ones((2, 3), dtype=torch.float32), torch.zeros(3, dtype=torch.float32)
    )
    sample64 = _make_sample(
        torch.ones((2, 3), dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    )
    with pytest.raises(ValueError):
        collate_list([sample32, sample64])


def test_collate_pad_mask_and_values():
    a = _make_sample(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0.0, 1.0]))
    b = _make_sample(torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]), torch.tensor([1.0, 0.0]))

    result = collate_pad([a, b])
    assert result["points"].shape == (2, 3, 2)
    assert result["direction"].shape == (2, 2)
    assert torch.all(result["points"][0, :2] == a.points)
    assert torch.all(result["points"][1, :3] == b.points)
    assert torch.all(result["mask"][0] == torch.tensor([True, True, False]))
    assert torch.all(result["mask"][1] == torch.tensor([True, True, True]))


def test_collate_pad_allows_zero_length_samples():
    sample = _make_sample(torch.empty((0, 3)), torch.zeros(3))
    result = collate_pad([sample])
    assert result["points"].shape == (1, 0, 3)
    assert result["mask"].shape == (1, 0)


def test_collate_pad_empty_batch_raises():
    with pytest.raises(ValueError):
        collate_pad([])


def test_collate_pad_dtype_mismatch_raises():
    sample32 = _make_sample(
        torch.ones((1, 2), dtype=torch.float32), torch.zeros(2, dtype=torch.float32)
    )
    sample64 = _make_sample(
        torch.ones((1, 2), dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    )
    with pytest.raises(ValueError):
        collate_pad([sample32, sample64])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_collate_pad_device_mismatch_raises():
    sample_cpu = _make_sample(torch.ones((1, 2)), torch.zeros(2))
    sample_cuda = _make_sample(
        torch.ones((1, 2), device=torch.device("cuda")),
        torch.zeros(2, device=torch.device("cuda")),
    )
    with pytest.raises(ValueError):
        collate_pad([sample_cpu, sample_cuda])
