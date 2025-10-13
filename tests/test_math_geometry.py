from __future__ import annotations

import math

import torch

from viterbo.math.geometry import (
    matmul_vertices,
    rotated_regular_ngon2d,
    translate_vertices,
    volume,
)


torch.set_default_dtype(torch.float64)


def test_matmul_and_translate_vertices() -> None:
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    transformed = matmul_vertices(matrix, vertices)
    expected = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    torch.testing.assert_close(transformed, expected)
    translation = torch.tensor([1.0, -1.0])
    translated = translate_vertices(translation, transformed)
    expected_translated = expected + translation
    torch.testing.assert_close(translated, expected_translated)


def test_volume_known_shapes() -> None:
    square = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    assert math.isclose(volume(square).item(), 1.0, rel_tol=1e-6, abs_tol=1e-6)
    cube = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert math.isclose(volume(cube).item(), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_rotated_regular_ngon() -> None:
    k = 6
    angle = math.pi / 6
    vertices, normals, offsets = rotated_regular_ngon2d(k, angle)
    assert vertices.shape == (k, 2)
    assert normals.shape == (k, 2)
    assert offsets.shape == (k,)
    # Check rotational symmetry: norms of offsets equal cos(pi/k)
    expected_offset = math.cos(math.pi / k)
    assert torch.allclose(offsets, torch.full_like(offsets, expected_offset), atol=1e-6)
