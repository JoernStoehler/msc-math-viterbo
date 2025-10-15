"""Common helpers for EHZ capacity solvers (planar + 4D utilities)."""

from __future__ import annotations

import torch

from viterbo.math.polytope import support


def point_exists(points: torch.Tensor, candidate: torch.Tensor, tol: float) -> bool:
    """Return True if ``candidate`` is within ``tol`` of any row of ``points``."""
    if points.size(0) == 0:
        return False
    diffs = (points - candidate).abs()
    return bool(torch.any(torch.all(diffs <= tol, dim=1)))


def split_lagrangian_product_vertices(
    vertices: torch.Tensor, tol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split 4D cartesian-structured vertices into planar factors ``(Q, P)``.

    Args:
      vertices: ``(M, 4)`` tensor of unique 4D vertices.
      tol: tolerance for membership checks when validating the cartesian product.

    Returns:
      ``(vertices_q, vertices_p)`` each shaped ``(Mq, 2)`` and ``(Mp, 2)``.

    Raises:
      NotImplementedError: if the vertices do not form a cartesian product.
    """
    if vertices.ndim != 2 or vertices.size(1) != 4:
        raise ValueError("vertices must be (M, 4) for Lagrangian product detection")
    unique_vertices = torch.unique(vertices, dim=0)
    vertices_q = torch.unique(unique_vertices[:, :2], dim=0)
    vertices_p = torch.unique(unique_vertices[:, 2:], dim=0)

    if vertices_q.size(0) < 3 or vertices_p.size(0) < 3:
        raise NotImplementedError("4D support currently expects at least polygons in both factors")

    expected = vertices_q.size(0) * vertices_p.size(0)
    if expected != unique_vertices.size(0):
        raise NotImplementedError(
            "4D support currently limited to Lagrangian products with a cartesian vertex structure"
        )

    for q_vertex in vertices_q:
        for p_vertex in vertices_p:
            combined = torch.cat((q_vertex, p_vertex))
            if not point_exists(unique_vertices, combined, tol):
                raise NotImplementedError(
                    "4D support currently limited to Lagrangian products with a cartesian vertex structure"
                )

    return vertices_q, vertices_p


def validate_planar_vertices(vertices: torch.Tensor, name: str) -> None:
    """Validate that ``vertices`` is a polygon ``(N, 2)`` with ``N ≥ 3``."""
    if vertices.ndim != 2 or vertices.size(1) != 2:
        raise ValueError(f"{name} must be a (N, 2) tensor")
    if vertices.size(0) < 3:
        raise ValueError(f"{name} must contain at least three vertices")


def validate_halfspaces_planar(
    normals: torch.Tensor, offsets: torch.Tensor, normals_name: str, offsets_name: str
) -> None:
    """Validate that halfspaces define a planar convex set with positive offsets."""
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError(f"{normals_name} must be (F, 2) and {offsets_name} must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError(f"{normals_name} and {offsets_name} must share the first dimension")
    if normals.size(1) != 2:
        raise ValueError(f"{normals_name} must describe planar halfspaces")
    if torch.any(offsets <= 0):
        raise ValueError(f"{offsets_name} must be strictly positive for a valid convex body")


def order_vertices_ccw(vertices: torch.Tensor) -> torch.Tensor:
    """Return the input polygon vertices ordered counter-clockwise."""
    if vertices.ndim != 2 or vertices.size(1) != 2:
        raise ValueError("vertices must be (M, 2) tensor")
    if vertices.size(0) < 3:
        raise ValueError("need at least three vertices for a 2D polygon")
    centroid = vertices.mean(dim=0)
    shifted = vertices - centroid
    angles = torch.atan2(shifted[:, 1], shifted[:, 0])
    order = torch.argsort(angles)
    ordered = vertices[order]
    return ordered


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """Return the positive oriented area of a planar polygon."""
    ordered = order_vertices_ccw(vertices)
    rolled = ordered.roll(shifts=-1, dims=0)
    cross = ordered[:, 0] * rolled[:, 1] - ordered[:, 1] * rolled[:, 0]
    area = 0.5 * torch.sum(cross)
    return area.abs()


def satisfies_reflection_at_vertex(
    vertices_q: torch.Tensor, direction: torch.Tensor, idx_vertex: int, tol: float
) -> bool:
    """Check strong reflection: vertex ``idx_vertex`` maximises ``<x, direction>``."""
    if torch.linalg.norm(direction) <= tol:
        return False
    support_value = support(vertices_q, direction)
    candidate_value = torch.dot(vertices_q[idx_vertex], direction)
    return bool(torch.isclose(candidate_value, support_value, atol=tol, rtol=0.0))
