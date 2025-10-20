from __future__ import annotations

import pytest
import torch
from tests.polytopes import PLANAR_POLYTOPE_PAIRS

from viterbo.math.capacity_ehz.stubs import oriented_edge_spectrum_4d
from viterbo.math.constructions import lagrangian_product, matmul_vertices

torch.set_default_dtype(torch.float64)


@pytest.mark.smoke
def test_oriented_edge_memo_smoke_on_product() -> None:
    """Memo/budgets with a small rotation cap returns a finite scalar on a tiny product."""
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    vertices, normals, offsets = lagrangian_product(square_q.vertices, square_p.vertices)
    cap = float(torch.pi)  # tighter than default to bound search
    out = oriented_edge_spectrum_4d(
        vertices,
        normals,
        offsets,
        rotation_cap=cap,
        use_cF_budgets=True,
        cF_constant=None,
        use_memo=True,
        memo_grid=1e-6,
        memo_buckets=16,
    )
    assert out.ndim == 0 and torch.isfinite(out)


@pytest.mark.smoke
def test_oriented_edge_memo_smoke_on_nonproduct() -> None:
    """Memo/budgets with a small rotation cap returns a finite scalar on a non-product."""
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    base_vertices, base_normals, base_offsets = lagrangian_product(
        square_q.vertices, square_p.vertices
    )
    # Break product structure with a gentle linear transform.
    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.0, 1.1, -0.3, 0.0],
            [0.1, 0.0, 1.0, 0.4],
            [0.0, -0.2, 0.0, 1.2],
        ],
        dtype=torch.get_default_dtype(),
    )
    vertices = matmul_vertices(matrix, base_vertices)
    cap = float(torch.pi)
    # Recompute H-rep after linear transform to preserve simplicity and consistency.
    from viterbo.math.polytope import vertices_to_halfspaces

    normals, offsets = vertices_to_halfspaces(vertices)
    out = oriented_edge_spectrum_4d(
        vertices,
        normals,
        offsets,
        rotation_cap=cap,
        use_cF_budgets=True,
        cF_constant=None,
        use_memo=True,
        memo_grid=1e-6,
        memo_buckets=16,
    )
    assert out.ndim == 0 and torch.isfinite(out)
