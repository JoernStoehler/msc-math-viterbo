from __future__ import annotations

import torch
from tests.polytopes import STANDARD_POLYTOPES_BY_NAME

from viterbo.math.constructions import (
    counterexample_pentagon_product,
    lagrangian_product,
    mixed_nonproduct_from_product,
    noisy_pentagon_product,
    random_polygon,
    random_polytope_algorithm1,
    random_polytope_algorithm2,
    regular_simplex,
    triangle_area_one,
    unit_square,
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


# ---- Canonical constructors -------------------------------------------------


def _unit_row_norms(normals: torch.Tensor, atol: float = 1e-8) -> None:
    norms = torch.linalg.norm(normals, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=atol, rtol=0.0)


def test_unit_square_shapes_and_area() -> None:
    v, n, c = unit_square()
    assert v.shape == (4, 2)
    assert n.shape[1] == 2 and c.shape[0] == n.shape[0]
    assert v.dtype == torch.float64 and n.dtype == torch.float64 and c.dtype == torch.float64
    _unit_row_norms(n)
    area = volume(v)
    torch.testing.assert_close(area, torch.tensor(4.0, dtype=torch.float64))


def test_triangle_area_one() -> None:
    v, n, c = triangle_area_one()
    assert v.shape == (3, 2)
    assert n.shape[1] == 2 and c.shape[0] == n.shape[0]
    _unit_row_norms(n)
    area = volume(v)
    torch.testing.assert_close(area, torch.tensor(1.0, dtype=torch.float64))


def test_regular_simplex_4d_volume() -> None:
    v, n, c = regular_simplex(4)
    assert v.shape == (5, 4)
    assert n.shape[1] == 4 and c.shape[0] == n.shape[0]
    _unit_row_norms(n, atol=1e-7)
    vol = volume(v)
    torch.testing.assert_close(
        vol, torch.tensor(1.0 / 24.0, dtype=torch.float64), atol=1e-10, rtol=0.0
    )


def test_counterexample_pentagon_product_shapes() -> None:
    v, n, c = counterexample_pentagon_product()
    assert v.shape == (25, 4)
    assert n.shape[1] == 4 and c.shape[0] == n.shape[0]
    _unit_row_norms(n, atol=1e-7)
    assert torch.all(c > 0)


def test_noisy_pentagon_product_deterministic() -> None:
    v1, n1, c1 = noisy_pentagon_product()
    v2, n2, c2 = noisy_pentagon_product()
    torch.testing.assert_close(v1, v2)
    torch.testing.assert_close(n1, n2)
    torch.testing.assert_close(c1, c2)
    assert v1.shape == (25, 4)


def test_mixed_nonproduct_breaks_block_structure() -> None:
    v, n, c = mixed_nonproduct_from_product()
    assert v.shape[1] == 4 and n.shape[1] == 4 and c.shape[0] == n.shape[0]
    # Check that some normal has non-trivial components in both q and p blocks
    q_norm = torch.linalg.norm(n[:, :2], dim=1)
    p_norm = torch.linalg.norm(n[:, 2:], dim=1)
    assert bool(torch.any((q_norm > 1e-6) & (p_norm > 1e-6)))


def test_random_polygon_seed41_k6_determinism_and_ccw() -> None:
    v1, n1, c1 = random_polygon(seed=41, k=6)
    v2, n2, c2 = random_polygon(seed=41, k=6)
    torch.testing.assert_close(v1, v2)
    torch.testing.assert_close(n1, n2)
    torch.testing.assert_close(c1, c2)
    # Area positive and vertices ordered by increasing angle (CCW)
    assert volume(v1).item() > 0
    angles = torch.atan2(v1[:, 1], v1[:, 0])
    assert torch.all(angles[1:] >= angles[:-1])
