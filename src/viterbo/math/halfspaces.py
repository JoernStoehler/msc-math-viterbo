"""Half-space (H-rep) utilities and conversions.

This module focuses on conversions between vertex and half-space representations
of convex polytopes and simple transformations applied in H-rep. All functions
are pure and torch-first (accept caller's device; no implicit moves).
"""

from __future__ import annotations

import itertools
from typing import Iterable

import torch


def _ensure_full_dimension(vertices: torch.Tensor) -> None:
    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D tensor")
    if vertices.size(0) <= vertices.size(1):
        raise ValueError("need at least D + 1 vertices for a full-dimensional hull")
    rank = torch.linalg.matrix_rank(vertices - vertices.mean(dim=0, keepdim=True))
    if int(rank.item()) != vertices.size(1):
        raise ValueError("vertices must span a full-dimensional polytope")


def _lexicographic_order(points: torch.Tensor) -> torch.Tensor:
    order = torch.arange(points.size(0))
    for dim in range(points.size(1) - 1, -1, -1):
        values = points[:, dim]
        order = order[torch.argsort(values[order])]
    return order


def _pairwise_unique(
    normals: Iterable[torch.Tensor],
    offsets: Iterable[torch.Tensor],
    tol: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    uniq_normals: list[torch.Tensor] = []
    uniq_offsets: list[torch.Tensor] = []
    for normal, offset in zip(normals, offsets):
        duplicate = False
        for existing_normal, existing_offset in zip(uniq_normals, uniq_offsets):
            if torch.allclose(normal, existing_normal, atol=tol, rtol=0.0) and torch.allclose(
                offset, existing_offset, atol=tol, rtol=0.0
            ):
                duplicate = True
                break
        if not duplicate:
            uniq_normals.append(normal)
            uniq_offsets.append(offset)
    return uniq_normals, uniq_offsets


def _facet_from_indices(
    vertices: torch.Tensor, indices: tuple[int, ...], centroid: torch.Tensor, tol: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    subset = vertices[list(indices)]
    base = subset[0]
    diffs = subset[1:] - base
    if diffs.size(0) == 0:
        return None
    rank = torch.linalg.matrix_rank(diffs)
    if int(rank.item()) != diffs.size(0):
        return None
    _, _, vh = torch.linalg.svd(diffs, full_matrices=True)
    normal = vh[-1]
    norm = torch.linalg.norm(normal)
    if norm <= tol:
        return None
    normal = normal / norm
    offset = torch.dot(normal, base)
    if (centroid @ normal) - offset > tol:
        normal = -normal
        offset = -offset
    support_values = vertices @ normal
    if torch.max(support_values - offset) > tol:
        return None
    if torch.sum(torch.isclose(support_values, offset, atol=tol, rtol=0.0)) < vertices.size(1):
        return None
    return normal, offset


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
    _ensure_full_dimension(vertices)
    dtype = vertices.dtype
    device = vertices.device
    dim = vertices.size(1)
    tol = max(float(torch.finfo(dtype).eps) ** 0.5, 1e-9)
    centroid = vertices.mean(dim=0)

    if dim == 1:
        min_val = vertices.min()
        max_val = vertices.max()
        normals = torch.tensor([[1.0], [-1.0]], dtype=dtype, device=device)
        offsets = torch.stack([max_val, -min_val])
        return normals, offsets

    candidate_normals: list[torch.Tensor] = []
    candidate_offsets: list[torch.Tensor] = []
    for indices in itertools.combinations(range(vertices.size(0)), dim):
        facet = _facet_from_indices(vertices, indices, centroid, tol)
        if facet is None:
            continue
        normal, offset = facet
        candidate_normals.append(normal)
        candidate_offsets.append(offset)

    if not candidate_normals:
        raise ValueError("failed to construct any supporting halfspaces")

    normals_unique, offsets_unique = _pairwise_unique(candidate_normals, candidate_offsets, tol)
    normals_tensor = torch.stack(normals_unique)
    offsets_tensor = torch.stack(offsets_unique)
    return normals_tensor.to(device=device, dtype=dtype), offsets_tensor.to(device=device, dtype=dtype)


def halfspaces_to_vertices(normals: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Convert half-space representation to vertices.

    Args:
      normals: (F, D) float tensor of facet normals (rows of ``B``).
      offsets: (F,) float tensor of facet offsets (``c``).

    Returns:
      vertices: (M, D) float tensor of vertices in V-rep order (implementation-defined).
    """
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,)")
    if normals.size(0) != offsets.size(0):
        raise ValueError("normals and offsets must have matching first dimension")
    dim = normals.size(1)
    if dim == 0:
        raise ValueError("dimension must be positive")
    tol = max(float(torch.finfo(normals.dtype).eps) ** 0.5, 1e-9)
    candidates: list[torch.Tensor] = []
    for indices in itertools.combinations(range(normals.size(0)), dim):
        sub_normals = normals[list(indices)]
        sub_offsets = offsets[list(indices)]
        try:
            vertex = torch.linalg.solve(sub_normals, sub_offsets)
        except RuntimeError:
            # Singular system -> skip
            continue
        if torch.max(normals @ vertex - offsets) > tol:
            continue
        if any(
            torch.allclose(vertex, existing, atol=tol, rtol=0.0)
            for existing in candidates
        ):
            continue
        candidates.append(vertex)
    if not candidates:
        raise ValueError("no feasible vertices found for the provided halfspaces")
    vertices_tensor = torch.stack(candidates)
    # Deterministic ordering via lexicographic sort on CPU, then move back
    cpu_vertices = vertices_tensor.detach().cpu()
    order = _lexicographic_order(cpu_vertices)
    ordered = vertices_tensor[order.to(vertices_tensor.device)]
    return ordered


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
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D square tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,)")
    if matrix.size(0) != matrix.size(1):
        raise ValueError("matrix must be square")
    if matrix.size(1) != normals.size(1):
        raise ValueError("matrix and normals dimensions are inconsistent")
    transformed_normals = torch.linalg.solve(matrix.T, normals.T).T
    norms = torch.linalg.norm(transformed_normals, dim=1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError("transformed normals contain zero norm rows")
    transformed_normals = transformed_normals / norms
    new_offsets = offsets / norms.squeeze(1)
    return transformed_normals, new_offsets


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
    if translation.ndim != 1:
        raise ValueError("translation must be a 1D tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,)")
    if translation.size(0) != normals.size(1):
        raise ValueError("translation and normals dimensions mismatch")
    new_offsets = offsets + normals @ translation
    return normals.clone(), new_offsets
