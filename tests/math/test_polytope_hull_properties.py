from __future__ import annotations

import pytest
import torch

from tests.polytopes import STANDARD_POLYTOPES_BY_NAME
from viterbo.math.polytope import halfspaces_to_vertices, vertices_to_halfspaces


def _sorted_rows(tensor: torch.Tensor) -> torch.Tensor:
    order = torch.arange(tensor.size(0), device=tensor.device)
    for dim in range(tensor.size(1) - 1, -1, -1):
        order = order[torch.argsort(tensor[order, dim])]
    return tensor[order]


def test_vertices_to_halfspaces_invariants_square() -> None:
    square = STANDARD_POLYTOPES_BY_NAME["square_2d"].vertices
    normals, offsets = vertices_to_halfspaces(square)
    # unit normals
    norms = torch.linalg.norm(normals, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-9, rtol=0.0)
    # all vertices satisfy inequalities
    lhs = square @ normals.T
    assert torch.all(lhs <= offsets + 1e-9)
    # round-trip recovers original vertices up to ordering
    rec_vertices = halfspaces_to_vertices(normals, offsets)
    torch.testing.assert_close(_sorted_rows(rec_vertices), _sorted_rows(square))


def test_vertices_to_halfspaces_invariants_hypercube_4d() -> None:
    hyper = STANDARD_POLYTOPES_BY_NAME["hypercube_4d_unit"].vertices
    normals, offsets = vertices_to_halfspaces(hyper)
    norms = torch.linalg.norm(normals, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-9, rtol=0.0)
    lhs = hyper @ normals.T
    assert torch.all(lhs <= offsets + 1e-9)
    rec_vertices = halfspaces_to_vertices(normals, offsets)
    torch.testing.assert_close(_sorted_rows(rec_vertices), _sorted_rows(hyper), atol=1e-6, rtol=0.0)


def test_vertices_to_halfspaces_rejects_lower_dimensional() -> None:
    # 3 points on a line in 2D -> not full dimensional polygon
    degenerate = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.get_default_dtype())
    with pytest.raises(ValueError):  # type: ignore[name-defined]
        vertices_to_halfspaces(degenerate)
