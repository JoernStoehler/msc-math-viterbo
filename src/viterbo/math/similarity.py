"""Similarity metrics between convex polytopes (stubs)."""

from __future__ import annotations

import torch


def hausdorff_distance(vertices_a: torch.Tensor, vertices_b: torch.Tensor) -> torch.Tensor:
    """Return the (symmetric) Hausdorff distance between two polytopes (stub)."""
    raise NotImplementedError


def hausdorff_distance_under_symplectic_group(
    vertices_a: torch.Tensor, vertices_b: torch.Tensor
) -> torch.Tensor:
    """Hausdorff distance up to symplectomorphisms (stub)."""
    raise NotImplementedError


def support_l2_distance(
    vertices_a: torch.Tensor, vertices_b: torch.Tensor, samples: int
) -> torch.Tensor:
    """Approximate L2 distance between support functions using random directions (stub)."""
    raise NotImplementedError
