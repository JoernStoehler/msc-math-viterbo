"""Symplectic geometry utilities and capacities (stubs).

This module hosts symplectic forms, random symplectic matrices, Lagrangian
products, and placeholders for EHZ capacity algorithms and minimal action
cycles. All functions are pure and torch-first.
"""

from __future__ import annotations

import torch

from .halfspaces import vertices_to_halfspaces


def symplectic_form(dimension: int) -> torch.Tensor:
    """Standard symplectic form matrix ``J`` of size ``(d, d)``.

    ``J = [[0, I], [-I, 0]]`` where ``d`` must be even.

    Args:
      dimension: even integer ``d``.

    Returns:
      J: (d, d) float tensor.
    """
    if dimension <= 0 or dimension % 2 != 0:
        raise ValueError("dimension must be a positive even integer")
    half = dimension // 2
    dtype = torch.get_default_dtype()
    eye = torch.eye(half, dtype=dtype)
    top = torch.cat([torch.zeros_like(eye), eye], dim=1)
    bottom = torch.cat([-eye, torch.zeros_like(eye)], dim=1)
    return torch.cat([top, bottom], dim=0)


def random_symplectic_matrix(dimension: int, seed: int | torch.Generator) -> torch.Tensor:
    """Random symplectic matrix ``M`` satisfying ``M.T @ J @ M = J``.

    Args:
      dimension: even integer ``d``.
      seed: Python int or ``torch.Generator``.

    Returns:
      M: (d, d) float tensor.
    """
    if dimension <= 0 or dimension % 2 != 0:
        raise ValueError("dimension must be a positive even integer")
    generator = torch.Generator(device="cpu")
    if isinstance(seed, torch.Generator):
        generator = seed
    else:
        generator.manual_seed(int(seed))
    dtype = torch.get_default_dtype()
    half = dimension // 2
    # Generate invertible block A via QR to ensure stability
    random_matrix = torch.randn((half, half), generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(random_matrix)
    a = q
    # Symmetric matrices for shear factors
    sym_upper = torch.randn((half, half), generator=generator, dtype=dtype)
    sym_upper = (sym_upper + sym_upper.T) / 2.0
    sym_lower = torch.randn((half, half), generator=generator, dtype=dtype)
    sym_lower = (sym_lower + sym_lower.T) / 2.0

    identity = torch.eye(half, dtype=dtype)
    block_a = torch.block_diag(a, torch.linalg.inv(a.T))
    upper = torch.block_diag(identity, identity)
    upper = upper.clone()
    upper[:half, half:] = sym_upper
    lower = torch.block_diag(identity, identity)
    lower = lower.clone()
    lower[half:, :half] = sym_lower
    matrix = upper @ block_a @ lower
    return matrix


def lagrangian_product(vertices_P: torch.Tensor, vertices_Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lagrangian product of two polytopes P, Q given their vertices.

    Args:
      vertices_P: (M1, d/2)
      vertices_Q: (M2, d/2)

    Returns:
      (vertices, normals, offsets) of the product polytope in ``R^d``.
    """
    if vertices_P.ndim != 2 or vertices_Q.ndim != 2:
        raise ValueError("vertices_P and vertices_Q must be 2D tensors")
    dim_p = vertices_P.size(1)
    dim_q = vertices_Q.size(1)
    if dim_p != dim_q:
        raise ValueError("vertices_P and vertices_Q must have matching dimensions")
    if vertices_P.device != vertices_Q.device or vertices_P.dtype != vertices_Q.dtype:
        raise ValueError("vertices_P and vertices_Q must share dtype and device")
    normals_p, offsets_p = vertices_to_halfspaces(vertices_P)
    normals_q, offsets_q = vertices_to_halfspaces(vertices_Q)
    dtype = vertices_P.dtype
    device = vertices_P.device
    zeros_p = torch.zeros((normals_p.size(0), dim_q), dtype=dtype, device=device)
    zeros_q = torch.zeros((normals_q.size(0), dim_p), dtype=dtype, device=device)
    normals = torch.cat(
        [torch.cat([normals_p, zeros_p], dim=1), torch.cat([zeros_q, normals_q], dim=1)], dim=0
    )
    offsets = torch.cat([offsets_p, offsets_q], dim=0)
    vp = vertices_P.unsqueeze(1).expand(-1, vertices_Q.size(0), -1)
    vq = vertices_Q.unsqueeze(0).expand(vertices_P.size(0), -1, -1)
    vertices = torch.cat([vp, vq], dim=2).reshape(-1, dim_p + dim_q)
    return vertices, normals, offsets


def capacity_ehz_algorithm1(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """EHZ capacity given H-rep (placeholder).

    Args:
      normals: (F, d)
      offsets: (F,)

    Returns:
      capacity: scalar float tensor.
    """
    raise NotImplementedError


def capacity_ehz_algorithm2(vertices: torch.Tensor) -> torch.Tensor:
    """EHZ capacity given V-rep (placeholder).

    Args:
      vertices: (M, d)

    Returns:
      capacity: scalar float tensor.
    """
    raise NotImplementedError


def minimal_action_cycle(
    vertices: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal action cycle (placeholder).

    Args:
      vertices: (M, d)
      normals: (F, d)
      offsets: (F,)

    Returns:
      (capacity, cycle): scalar capacity and (K, d) cycle points.
    """
    raise NotImplementedError


def systolic_ratio(volume: torch.Tensor, capacity_ehz: torch.Tensor) -> torch.Tensor:
    """Systolic ratio ``vol / capacity^{n}`` (definition TBD).

    Args:
      volume: scalar float tensor.
      capacity_ehz: scalar float tensor.

    Returns:
      ratio: scalar float tensor.
    """
    raise NotImplementedError

