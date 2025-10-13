from __future__ import annotations

import torch

from viterbo.math.symplectic import (
    lagrangian_product,
    random_symplectic_matrix,
    symplectic_form,
)


torch.set_default_dtype(torch.float64)


def _sorted_rows(tensor: torch.Tensor) -> torch.Tensor:
    order = torch.arange(tensor.size(0), device=tensor.device)
    for dim in range(tensor.size(1) - 1, -1, -1):
        order = order[torch.argsort(tensor[order, dim])]
    return tensor[order]


def test_symplectic_form_structure() -> None:
    j = symplectic_form(4)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(j, expected)


def test_random_symplectic_matrix_preserves_form() -> None:
    dimension = 4
    seed = 42
    matrix = random_symplectic_matrix(dimension, seed)
    j = symplectic_form(dimension)
    lhs = matrix.T @ j @ matrix
    torch.testing.assert_close(lhs, j, atol=1e-6, rtol=1e-6)


def test_lagrangian_product_block_structure() -> None:
    vertices_p = torch.tensor([[-1.0], [1.0]])
    vertices_q = torch.tensor([[0.0], [2.0]])
    vertices, normals, offsets = lagrangian_product(vertices_p, vertices_q)
    expected_vertices = torch.tensor(
        [[-1.0, 0.0], [-1.0, 2.0], [1.0, 0.0], [1.0, 2.0]]
    )
    torch.testing.assert_close(vertices, expected_vertices)
    # Halfspaces correspond to |x| <= 1 and 0 <= y <= 2
    expected_normals = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    expected_offsets = torch.tensor([1.0, 1.0, 2.0, 0.0])
    torch.testing.assert_close(_sorted_rows(normals), _sorted_rows(expected_normals), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        offsets.sort().values,
        expected_offsets.sort().values,
        atol=1e-6,
        rtol=1e-6,
    )
