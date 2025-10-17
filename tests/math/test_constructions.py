from __future__ import annotations

import torch
from tests.polytopes import STANDARD_POLYTOPES_BY_NAME

from viterbo.math.constructions import (
    lagrangian_product,
    random_polytope_algorithm1,
    random_polytope_algorithm2,
)
from viterbo.math.volume import volume

torch.set_default_dtype(torch.float64)


def _feasible(normals: torch.Tensor, offsets: torch.Tensor, vertices: torch.Tensor) -> bool:
    return torch.all((vertices @ normals.T) <= offsets + 1e-6)


def _sorted_rows(tensor: torch.Tensor) -> torch.Tensor:
    order = torch.arange(tensor.size(0), device=tensor.device)
    for dim in range(tensor.size(1) - 1, -1, -1):
        order = order[torch.argsort(tensor[order, dim])]
    return tensor[order]


def test_lagrangian_product_block_structure() -> None:
    segment_p = STANDARD_POLYTOPES_BY_NAME["segment_1d_symmetric_unit"]
    segment_q = STANDARD_POLYTOPES_BY_NAME["segment_1d_shifted_length2"]
    vertices, normals, offsets = lagrangian_product(segment_p.vertices, segment_q.vertices)

    expected_vertices = torch.cartesian_prod(
        segment_p.vertices.squeeze(1), segment_q.vertices.squeeze(1)
    ).to(segment_p.vertices.dtype)
    torch.testing.assert_close(vertices, expected_vertices)

    zeros_p = torch.zeros_like(segment_p.vertices)
    zeros_q = torch.zeros_like(segment_q.vertices)
    expected_normals = torch.cat(
        (
            torch.cat((segment_p.normals, zeros_p), dim=1),
            torch.cat((zeros_q, segment_q.normals), dim=1),
        ),
        dim=0,
    )
    expected_offsets = torch.cat((segment_p.offsets, segment_q.offsets))
    torch.testing.assert_close(
        _sorted_rows(normals), _sorted_rows(expected_normals), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        offsets.sort().values,
        expected_offsets.sort().values,
        atol=1e-6,
        rtol=1e-6,
    )


def test_random_polytope_algorithm1_deterministic_and_feasible() -> None:
    seed = 1234
    v1, n1, c1 = random_polytope_algorithm1(seed, num_facets=16, dimension=3)
    v2, n2, c2 = random_polytope_algorithm1(seed, num_facets=16, dimension=3)
    torch.testing.assert_close(v1, v2)
    torch.testing.assert_close(n1, n2)
    torch.testing.assert_close(c1, c2)
    assert v1.size(0) >= 4
    assert _feasible(n1, c1, v1)


def test_random_polytope_algorithm2_returns_convex_hull() -> None:
    seed = 5678
    vertices, normals, offsets = random_polytope_algorithm2(seed, num_vertices=12, dimension=2)
    assert vertices.ndim == 2
    assert normals.ndim == 2
    assert offsets.ndim == 1
    assert vertices.size(1) == 2
    assert normals.size(1) == 2
    assert _feasible(normals, offsets, vertices)
    v_again, n_again, c_again = random_polytope_algorithm2(seed, num_vertices=12, dimension=2)
    torch.testing.assert_close(vertices, v_again)
    torch.testing.assert_close(normals, n_again)
    torch.testing.assert_close(offsets, c_again)


def test_random_polytope_algorithm2_dimension4_roundtrip() -> None:
    seed = 1357
    vertices, normals, offsets = random_polytope_algorithm2(seed, num_vertices=8, dimension=4)
    assert vertices.ndim == 2 and vertices.size(1) == 4
    assert normals.ndim == 2 and normals.size(1) == 4
    assert offsets.ndim == 1
    assert _feasible(normals, offsets, vertices)
    recon_vertices, recon_normals, recon_offsets = random_polytope_algorithm2(
        seed, num_vertices=8, dimension=4
    )
    torch.testing.assert_close(vertices, recon_vertices)
    torch.testing.assert_close(normals, recon_normals)
    torch.testing.assert_close(offsets, recon_offsets)


def test_lagrangian_product_2d_blocks_form_4d_polytope() -> None:
    square = STANDARD_POLYTOPES_BY_NAME["square_2d"].vertices
    vertices, normals, offsets = lagrangian_product(square, square)
    assert vertices.shape == (square.size(0) ** 2, 4)
    assert normals.shape[1] == 4
    assert offsets.shape[0] == normals.shape[0]
    assert _feasible(normals, offsets, vertices)
    assert volume(vertices).item() > 0
