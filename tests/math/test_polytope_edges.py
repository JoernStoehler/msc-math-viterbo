from __future__ import annotations

import pytest
import torch

import viterbo.math.polytope as P
from viterbo.math.polytope import (
    bounding_box,
    facet_vertex_incidence,
    halfspace_violations,
    halfspaces_to_vertices,
    support,
    support_argmax,
    vertices_to_halfspaces,
)

torch.set_default_dtype(torch.float64)

# Include in smoke-tier runs
pytestmark = pytest.mark.smoke


def test_vertices_to_halfspaces_degenerate_2d_raises() -> None:
    # Fewer than D+1 vertices in 2D
    with pytest.raises(ValueError):
        vertices_to_halfspaces(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))

    # Colinear points in 2D (rank deficient)
    with pytest.raises(ValueError):
        vertices_to_halfspaces(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))

    # Wrong ndim
    with pytest.raises(ValueError):
        vertices_to_halfspaces(torch.tensor([0.0, 1.0]))


def test_halfspaces_to_vertices_dimension_mismatch_errors() -> None:
    # normals/offsets rank requirements
    with pytest.raises(ValueError):
        halfspaces_to_vertices(torch.tensor([1.0, 0.0]), torch.tensor([1.0]))

    # First dimension mismatch
    with pytest.raises(ValueError):
        halfspaces_to_vertices(torch.randn(3, 2), torch.randn(2))

    # Zero dimension (D == 0)
    with pytest.raises(ValueError):
        halfspaces_to_vertices(torch.empty(1, 0), torch.randn(1))


def test_halfspaces_to_vertices_infeasible_raises() -> None:
    # 1D contradictory halfspaces: x <= 0 and x >= 2  (encoded as -x <= -2)
    normals = torch.tensor([[1.0], [-1.0]])
    offsets = torch.tensor([0.0, -2.0])
    with pytest.raises(ValueError, match="no feasible vertices"):
        halfspaces_to_vertices(normals, offsets)


def test_bounding_box_edge_and_shape_cases() -> None:
    # Single point: min == max == point
    p = torch.tensor([[2.5, -3.0, 0.0]])
    mins, maxs = bounding_box(p)
    torch.testing.assert_close(mins, p[0])
    torch.testing.assert_close(maxs, p[0])

    # Empty set of points: torch reduction should raise
    empty = torch.empty((0, 2), dtype=torch.get_default_dtype())
    # PyTorch raises IndexError for zero-length reductions with dim specified
    with pytest.raises((RuntimeError, IndexError)):
        _ = bounding_box(empty)


def test_halfspace_violations_edge_and_shape_cases() -> None:
    # On-boundary points produce zero violations
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    offsets = torch.tensor([1.0, 2.0])
    points = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    viol = halfspace_violations(points, normals, offsets)
    torch.testing.assert_close(viol, torch.zeros_like(viol))

    # Dimension mismatch between points (N,3) and normals (F,2) should error
    with pytest.raises(RuntimeError):
        _ = halfspace_violations(torch.randn(4, 3), torch.randn(5, 2), torch.randn(5))

    # Offsets with wrong rank should trigger broadcast shape error
    with pytest.raises(RuntimeError):
        _ = halfspace_violations(torch.randn(3, 2), torch.randn(4, 2), torch.randn(4, 2))


def test_support_tie_and_near_tie_behavior() -> None:
    # Exact tie: expect first index selected by torch.max
    pts = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    direction = torch.tensor([1.0, 0.0])
    val, idx = support_argmax(pts, direction)
    torch.testing.assert_close(val, torch.tensor(1.0))
    assert idx == 0

    # Near-tie: second point wins by a tiny margin around sqrt(eps)
    eps = float(torch.finfo(torch.get_default_dtype()).eps) ** 0.5
    delta = eps * 0.5
    pts2 = torch.tensor([[1.0, 0.0], [1.0 + delta, 0.0]])
    v = support(pts2, direction)
    v2, idx2 = support_argmax(pts2, direction)
    torch.testing.assert_close(v, torch.tensor(1.0 + delta))
    torch.testing.assert_close(v2, torch.tensor(1.0 + delta))
    assert idx2 == 1


def test_facet_vertex_incidence_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        facet_vertex_incidence(torch.randn(4, 2), torch.randn(3, 2), torch.randn(3))


def test__facet_from_indices_triangle_edge_success() -> None:
    # Simple triangle in 2D; any edge should define a valid supporting facet.
    verts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    centroid = verts.mean(dim=0)
    tol = float(torch.finfo(verts.dtype).eps) ** 0.5
    res = P._facet_from_indices(verts, (0, 1), centroid, tol)
    assert res is not None
    normal, offset = res
    # Edge (0,1) lies on y=0; outward normal should point to decreasing y
    assert normal.shape == torch.Size([2])
    assert isinstance(offset.item(), float)
    # All vertices satisfy n·x <= c up to tol
    assert torch.max(verts @ normal - offset) <= tol + 1e-12


def test__facet_from_indices_single_index_none() -> None:
    verts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    centroid = verts.mean(dim=0)
    tol = float(torch.finfo(verts.dtype).eps) ** 0.5
    assert P._facet_from_indices(verts, (0,), centroid, tol) is None
