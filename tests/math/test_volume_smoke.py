from __future__ import annotations

import math

import torch

from viterbo.math.volume import volume

torch.set_default_dtype(torch.float64)


def test_volume_one_dimensional_segment() -> None:
    vertices = torch.tensor([[-2.0], [3.5], [1.0]])
    length = volume(vertices)
    torch.testing.assert_close(length, torch.tensor(5.5, dtype=vertices.dtype))


def test_volume_triangle_matches_half_area_formula() -> None:
    triangle = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    area = volume(triangle)
    torch.testing.assert_close(area, torch.tensor(1.0, dtype=triangle.dtype))


def test_volume_tetrahedron_matches_determinant_formula() -> None:
    tetra = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    vol = volume(tetra)
    expected = torch.tensor((1.0 * 2.0 * 3.0) / 6.0, dtype=tetra.dtype)
    torch.testing.assert_close(vol, expected, atol=1e-7, rtol=0.0)


def test_volume_respects_input_dtype() -> None:
    triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    area = volume(triangle)
    assert area.dtype == torch.float32
    assert math.isclose(float(area), 1.0, rel_tol=1e-6, abs_tol=1e-6)
