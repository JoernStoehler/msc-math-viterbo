from __future__ import annotations

import torch

from viterbo.math.random_polytopes import (
    random_polytope_algorithm1,
    random_polytope_algorithm2,
)


torch.set_default_dtype(torch.float64)


def _feasible(normals: torch.Tensor, offsets: torch.Tensor, vertices: torch.Tensor) -> bool:
    return torch.all((vertices @ normals.T) <= offsets + 1e-6)


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
