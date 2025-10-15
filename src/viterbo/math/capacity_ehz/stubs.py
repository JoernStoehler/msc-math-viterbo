"""Stubs and planned solvers for EHZ capacity in higher dimensions."""

from __future__ import annotations

import torch


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
