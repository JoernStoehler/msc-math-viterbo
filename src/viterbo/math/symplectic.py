"""Symplectic geometry utilities and capacities (stubs).

This module hosts symplectic forms, random symplectic matrices, Lagrangian
products, and placeholders for EHZ capacity algorithms and minimal action
cycles. All functions are pure and torch-first.
"""

from __future__ import annotations

import torch


def symplectic_form(dimension: int) -> torch.Tensor:
    """Standard symplectic form matrix ``J`` of size ``(d, d)``.

    ``J = [[0, I], [-I, 0]]`` where ``d`` must be even.

    Args:
      dimension: even integer ``d``.

    Returns:
      J: (d, d) float tensor.
    """
    raise NotImplementedError


def random_symplectic_matrix(dimension: int, seed: int | torch.Generator) -> torch.Tensor:
    """Random symplectic matrix ``M`` satisfying ``M.T @ J @ M = J``.

    Args:
      dimension: even integer ``d``.
      seed: Python int or ``torch.Generator``.

    Returns:
      M: (d, d) float tensor.
    """
    raise NotImplementedError


def lagrangian_product(vertices_P: torch.Tensor, vertices_Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lagrangian product of two polytopes P, Q given their vertices.

    Args:
      vertices_P: (M1, d/2)
      vertices_Q: (M2, d/2)

    Returns:
      (vertices, normals, offsets) of the product polytope in ``R^d``.
    """
    raise NotImplementedError


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

