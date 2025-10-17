from __future__ import annotations

import math

import torch

from tests.polytopes import STANDARD_POLYTOPES_BY_NAME
from viterbo.math.volume import volume

torch.set_default_dtype(torch.float64)


def test_volume_one_dimensional_segment() -> None:
    segment = STANDARD_POLYTOPES_BY_NAME["segment_1d_neg2_3p5"]
    length = volume(segment.vertices)
    torch.testing.assert_close(length, segment.volume)


def test_volume_triangle_matches_half_area_formula() -> None:
    triangle = STANDARD_POLYTOPES_BY_NAME["right_triangle_area_one"].vertices
    area = volume(triangle)
    torch.testing.assert_close(area, torch.tensor(1.0, dtype=triangle.dtype))


def test_volume_tetrahedron_matches_determinant_formula() -> None:
    tetra = STANDARD_POLYTOPES_BY_NAME["tetrahedron_box_123"].vertices
    vol = volume(tetra)
    expected = torch.tensor((1.0 * 2.0 * 3.0) / 6.0, dtype=tetra.dtype)
    torch.testing.assert_close(vol, expected, atol=1e-7, rtol=0.0)


def test_volume_respects_input_dtype() -> None:
    triangle = STANDARD_POLYTOPES_BY_NAME["right_triangle_area_one"].vertices.to(torch.float32)
    area = volume(triangle)
    assert area.dtype == torch.float32
    assert math.isclose(float(area), 1.0, rel_tol=1e-6, abs_tol=1e-6)
