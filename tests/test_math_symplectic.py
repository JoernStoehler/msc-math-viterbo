from __future__ import annotations

import math

import torch

from viterbo.math.symplectic import (
    capacity_ehz_algorithm1,
    capacity_ehz_algorithm2,
    capacity_ehz_primal_dual,
    lagrangian_product,
    minimal_action_cycle,
    random_symplectic_matrix,
    symplectic_form,
    systolic_ratio,
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


def _square_halfspaces() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    return vertices_to_halfspaces(vertices)


def test_capacity_ehz_algorithm1_matches_polygon_area() -> None:
    normals, offsets = _square_halfspaces()
    capacity = capacity_ehz_algorithm1(normals, offsets)
    torch.testing.assert_close(capacity, torch.tensor(4.0))


def test_capacity_ehz_algorithm2_agrees_with_halfspace_solver() -> None:
    vertices = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    capacity_v = capacity_ehz_algorithm2(vertices)
    capacity_h = capacity_ehz_algorithm1(normals, offsets)
    torch.testing.assert_close(capacity_v, capacity_h)


def test_capacity_ehz_primal_dual_validates_consistency() -> None:
    vertices = torch.tensor(
        [
            [-0.5, -1.0],
            [-0.5, 1.0],
            [0.5, -1.0],
            [0.5, 1.0],
        ]
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    capacity = capacity_ehz_primal_dual(vertices, normals, offsets)
    torch.testing.assert_close(capacity, torch.tensor(4.0 * 0.5))


def test_minimal_action_cycle_returns_ordered_boundary() -> None:
    vertices = torch.tensor(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [2.0, -1.0],
            [2.0, 1.0],
        ]
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    capacity, cycle = minimal_action_cycle(vertices, normals, offsets)
    torch.testing.assert_close(capacity, torch.tensor(8.0))
    assert cycle.size(0) == vertices.size(0)
    # Ensure counter-clockwise order by checking signed area is positive
    rolled = cycle.roll(-1, dims=0)
    signed_area = 0.5 * torch.sum(cycle[:, 0] * rolled[:, 1] - cycle[:, 1] * rolled[:, 0])
    assert signed_area > 0


def test_systolic_ratio_matches_ball_normalisation() -> None:
    r = torch.tensor(1.0)
    # 2D ball (n = 1)
    volume_2d = math.pi * r**2
    capacity_2d = math.pi * r**2
    ratio_2d = systolic_ratio(torch.tensor(volume_2d), torch.tensor(capacity_2d), 2)
    torch.testing.assert_close(ratio_2d, torch.tensor(1.0))

    # 4D ball (n = 2)
    volume_4d = (math.pi**2) * r**4 / 2.0
    capacity_4d = math.pi * r**2
    ratio_4d = systolic_ratio(torch.tensor(volume_4d), torch.tensor(capacity_4d), 4)
    torch.testing.assert_close(ratio_4d, torch.tensor(2.0))
