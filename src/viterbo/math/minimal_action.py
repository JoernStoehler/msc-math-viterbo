"""EHZ capacity and minimal action cycles (4D focus, stubs included)."""

from __future__ import annotations

import torch

from viterbo.math.polytope import halfspaces_to_vertices


def capacity_ehz_algorithm1(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""Ekeland–Hofer–Zehnder capacity via the Artstein–Avidan–Ostrover program (2D placeholder)."""
    _ensure_planar(normals, offsets)
    vertices = halfspaces_to_vertices(normals, offsets)
    return _polygon_area(vertices)


def capacity_ehz_algorithm2(vertices: torch.Tensor) -> torch.Tensor:
    r"""EHZ capacity via discrete billiards on vertices (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    d = vertices.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d != 2:
        raise NotImplementedError(
            "capacity_ehz_algorithm2 currently supports 2D only; 4D support is planned"
        )
    if vertices.size(0) < 3:
        raise ValueError("need at least three vertices for a 2D polygon")
    return _polygon_area(vertices)


def capacity_ehz_primal_dual(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    r"""Hybrid primal–dual EHZ capacity solver (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(1) != vertices.size(1):
        raise ValueError("vertices and normals must share the same ambient dimension")
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d != 2:
        raise NotImplementedError(
            "capacity_ehz_primal_dual currently supports 2D only; 4D support is planned"
        )
    capacity = capacity_ehz_algorithm2(vertices)
    reference = capacity_ehz_algorithm1(normals, offsets)
    if not torch.allclose(capacity, reference, atol=1e-8, rtol=1e-8):
        raise ValueError("inconsistent primal and dual capacities for the provided polygon")
    return capacity


def minimal_action_cycle(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the minimal action Reeb orbit (2D placeholder)."""
    if vertices.ndim != 2:
        raise ValueError("vertices must be a (M, d) tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if vertices.size(1) != normals.size(1):
        raise ValueError("vertices and normals must share the same ambient dimension")
    d = vertices.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    if d != 2:
        raise NotImplementedError("minimal_action_cycle currently supports 2D only; 4D planned")
    ordered_vertices = _order_vertices_counter_clockwise(vertices)
    capacity = _polygon_area(ordered_vertices)
    return capacity, ordered_vertices


def systolic_ratio(
    volume: torch.Tensor, capacity_ehz: torch.Tensor, symplectic_dimension: int | None = None
) -> torch.Tensor:
    r"""Viterbo systolic ratio ``vol(K) / c_{EHZ}(K)^{n}`` for ``2n``-dimensional bodies."""
    if volume.ndim != 0 or capacity_ehz.ndim != 0:
        raise ValueError("volume and capacity_ehz must be scalar tensors")
    if torch.any(capacity_ehz <= 0):
        raise ValueError("capacity_ehz must be strictly positive")
    if symplectic_dimension is None:
        raise ValueError("symplectic_dimension must be provided for systolic_ratio")
    if symplectic_dimension % 2 != 0 or symplectic_dimension <= 0:
        raise ValueError("symplectic_dimension must be a positive even integer")
    n = symplectic_dimension // 2
    return volume / capacity_ehz.pow(n)


def _ensure_planar(normals: torch.Tensor, offsets: torch.Tensor) -> None:
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must have matching first dimension")
    if normals.size(1) != 2:
        raise NotImplementedError(
            "capacity_ehz_algorithm1 currently supports planar polytopes only"
        )


def _order_vertices_counter_clockwise(vertices: torch.Tensor) -> torch.Tensor:
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


def _polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    ordered = _order_vertices_counter_clockwise(vertices)
    rolled = ordered.roll(shifts=-1, dims=0)
    cross = ordered[:, 0] * rolled[:, 1] - ordered[:, 1] * rolled[:, 0]
    area = 0.5 * torch.sum(cross)
    return area.abs()


# 4D-focused stubs -------------------------------------------------------------


def capacity_ehz_haim_kislev(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""General EHZ formula (Haim–Kislev) operating on the H-representation (stub)."""
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    raise NotImplementedError


def oriented_edge_spectrum_4d(
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
    *,
    k_max: int | None = None,
) -> torch.Tensor:
    r"""Hutchings-style oriented-edge action spectrum in R^4 (stub)."""
    if vertices.size(1) != 4:
        raise ValueError("oriented_edge_spectrum_4d expects vertices in R^4")
    raise NotImplementedError


def capacity_ehz_via_qp(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""Facet-multiplier convex QP with QR-reduced constraints (stub)."""
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    raise NotImplementedError


def capacity_ehz_via_lp(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    r"""LP/SOCP relaxations (Krupp-style) providing bounds/warm-starts (stub)."""
    d = normals.size(1)
    if d % 2 != 0:
        raise ValueError("ambient dimension must be even (2n) for symplectic problems")
    raise NotImplementedError
