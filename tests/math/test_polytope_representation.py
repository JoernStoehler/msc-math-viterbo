from __future__ import annotations

import torch

from viterbo.math.constructions import (
    matmul_halfspace,
    matmul_vertices,
    translate_halfspace,
    translate_vertices,
)
from viterbo.math.polytope import halfspaces_to_vertices, vertices_to_halfspaces

torch.set_default_dtype(torch.float64)


def _sorted_rows(tensor: torch.Tensor) -> torch.Tensor:
    order = torch.arange(tensor.size(0), device=tensor.device)
    for dim in range(tensor.size(1) - 1, -1, -1):
        order = order[torch.argsort(tensor[order, dim])]
    return tensor[order]


def test_vertices_halfspaces_roundtrip_cube() -> None:
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    reconstructed = halfspaces_to_vertices(normals, offsets)
    torch.testing.assert_close(
        _sorted_rows(vertices),
        _sorted_rows(reconstructed),
        atol=1e-6,
        rtol=0.0,
    )


def test_halfspace_transforms_agree_with_vertices() -> None:
    square = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    normals, offsets = vertices_to_halfspaces(square)
    matrix = torch.tensor([[2.0, 1.0], [0.5, 3.0]])
    transformed_vertices = matmul_vertices(matrix, square)
    expected_normals, expected_offsets = vertices_to_halfspaces(transformed_vertices)
    actual_normals, actual_offsets = matmul_halfspace(matrix, normals, offsets)
    torch.testing.assert_close(
        _sorted_rows(expected_normals),
        _sorted_rows(actual_normals),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        expected_offsets.sort().values,
        actual_offsets.sort().values,
        atol=1e-6,
        rtol=0.0,
    )
    translation = torch.tensor([0.5, -2.0])
    translated_vertices = translate_vertices(translation, transformed_vertices)
    expected_normals, expected_offsets = vertices_to_halfspaces(translated_vertices)
    actual_normals, actual_offsets = translate_halfspace(
        translation, actual_normals, actual_offsets
    )
    torch.testing.assert_close(
        _sorted_rows(expected_normals),
        _sorted_rows(actual_normals),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        expected_offsets.sort().values,
        actual_offsets.sort().values,
        atol=1e-6,
        rtol=0.0,
    )
