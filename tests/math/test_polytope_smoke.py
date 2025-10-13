from __future__ import annotations

import torch

from viterbo.math.polytope import (
    bounding_box,
    halfspace_violations,
    pairwise_squared_distances,
    support,
    support_argmax,
)

torch.set_default_dtype(torch.float64)


def test_support_scalar_dtype_and_value() -> None:
    points = torch.tensor([[0.0, 0.0], [2.0, -1.0], [1.0, 3.0]])
    direction = torch.tensor([1.0, 2.0])
    value = support(points, direction)
    assert value.shape == torch.Size([])
    assert value.dtype == points.dtype
    assert value.device == points.device
    torch.testing.assert_close(value, torch.tensor(7.0, dtype=points.dtype))


def test_support_argmax_returns_python_index() -> None:
    points = torch.tensor([[-1.0, 0.0], [2.0, -2.0], [0.5, 1.0]])
    direction = torch.tensor([-1.0, 2.0])
    value, idx = support_argmax(points, direction)
    assert isinstance(idx, int)
    torch.testing.assert_close(value, torch.tensor(1.5, dtype=points.dtype))
    assert idx == 2


def test_pairwise_squared_distances_is_symmetric() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    d2 = pairwise_squared_distances(points)
    torch.testing.assert_close(d2, d2.T)
    expected = torch.tensor([[0.0, 1.0, 4.0], [1.0, 0.0, 5.0], [4.0, 5.0, 0.0]])
    torch.testing.assert_close(d2, expected, atol=1e-7, rtol=0.0)


def test_halfspace_violations_detects_outside_points() -> None:
    normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    offsets = torch.tensor([1.0, 1.0, 1.0, 1.0])
    points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.5, -1.5]])
    violations = halfspace_violations(points, normals, offsets)
    torch.testing.assert_close(violations[0], torch.zeros_like(violations[0]))
    assert torch.isclose(violations[1, 0], torch.tensor(1.0))
    assert torch.isclose(violations[2, 3], torch.tensor(0.5))


def test_bounding_box_returns_min_max() -> None:
    points = torch.tensor([[1.0, -2.0, 0.5], [3.0, 1.0, -1.5], [-4.0, 0.0, 2.0]])
    mins, maxs = bounding_box(points)
    torch.testing.assert_close(mins, torch.tensor([-4.0, -2.0, -1.5], dtype=points.dtype))
    torch.testing.assert_close(maxs, torch.tensor([3.0, 1.0, 2.0], dtype=points.dtype))
