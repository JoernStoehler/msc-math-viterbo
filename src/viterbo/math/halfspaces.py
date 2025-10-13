"""Half-space (H-rep) utilities and conversions.

This module focuses on conversions between vertex and half-space representations
of convex polytopes and simple transformations applied in H-rep. All functions
are pure and torch-first (accept caller's device; no implicit moves).
"""

from __future__ import annotations

import torch


def vertices_to_halfspaces(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert vertices to half-space representation ``Bx <= c``.

    Args:
      vertices: (M, D) float tensor of vertices.

    Returns:
      (normals, offsets):
        - normals: (F, D) float tensor of facet normals (rows of ``B``).
        - offsets: (F,) float tensor of facet offsets (``c``).

    Notes:
      - Dtype/device follow ``vertices``; no implicit casts or moves.
      - Robustness/degeneracy handling is implementation-defined.
    """
    raise NotImplementedError


def halfspaces_to_vertices(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Convert half-space representation to vertices.

    Args:
      normals: (F, D) float tensor of facet normals (rows of ``B``).
      offsets: (F,) float tensor of facet offsets (``c``).

    Returns:
      vertices: (M, D) float tensor of vertices in V-rep order (implementation-defined).
    """
    raise NotImplementedError


def matmul_halfspace(
    matrix: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply linear map ``x -> A x`` to an H-rep ``Bx <= c``.

    The image polytope is ``{Ax | Bx <= c}``. Its H-rep can be expressed via
    transformed normals/offsets.

    Args:
      matrix: (D, D) float tensor ``A``.
      normals: (F, D) float tensor ``B``.
      offsets: (F,) float tensor ``c``.

    Returns:
      (new_normals, new_offsets): transformed H-rep with shapes matching inputs.
    """
    raise NotImplementedError


def translate_halfspace(
    translation: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate an H-rep polytope by ``t``.

    For ``P = {x | Bx <= c}``, ``P + t = {y | B(y - t) <= c}`` which yields
    new offsets ``c' = c + B t`` and unchanged normals.

    Args:
      translation: (D,) float tensor ``t``.
      normals: (F, D) float tensor ``B``.
      offsets: (F,) float tensor ``c``.

    Returns:
      (new_normals, new_offsets): ``(B, c + B t)`` with broadcasting rules applied.
    """
    raise NotImplementedError
