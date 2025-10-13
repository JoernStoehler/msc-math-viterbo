"""Geometric constructions of simple polytopes and configurations.

Torch-first helpers that return vertex and/or halfspace representations.
"""

from __future__ import annotations

import math

import torch

from viterbo.math.polytope import halfspaces_to_vertices, vertices_to_halfspaces


def _make_generator(seed: int | torch.Generator) -> torch.Generator:
    if isinstance(seed, torch.Generator):
        return seed
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _sample_in_unit_ball(generator: torch.Generator, count: int, dimension: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    points = torch.randn((count, dimension), generator=generator)
    norms = torch.linalg.norm(points, dim=1, keepdim=True)
    directions = points / norms
    radii = torch.rand((count, 1), generator=generator) ** (1.0 / dimension)
    return directions * radii


def rotated_regular_ngon2d(k: int, angle: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated regular n-gon in 2D.

    Args:
      k: number of sides ``k >= 3``.
      angle: rotation angle in radians.

    Returns:
      (vertices, normals, offsets):
        - vertices: (k, 2)
        - normals: (k, 2)
        - offsets: (k,)
    """
    if k < 3:
        raise ValueError("k must be at least 3")
    dtype = torch.get_default_dtype()
    device = torch.device("cpu")
    angles = (torch.arange(k, dtype=dtype, device=device) * (2.0 * math.pi / k)) + angle
    vertices = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    normals = []
    offsets = []
    for i in range(k):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % k]
        edge = v1 - v0
        normal = torch.tensor([edge[1], -edge[0]], dtype=dtype, device=device)
        normal = normal / torch.linalg.norm(normal)
        offset = torch.dot(normal, v0)
        if offset < 0:
            normal = -normal
            offset = -offset
        normals.append(normal)
        offsets.append(offset)
    normals_tensor = torch.stack(normals)
    offsets_tensor = torch.stack(offsets)
    return vertices, normals_tensor, offsets_tensor


def matmul_vertices(matrix: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Apply linear map ``x -> A x`` to a vertex set."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D")
    if matrix.size(0) != matrix.size(1) or matrix.size(1) != vertices.size(1):
        raise ValueError("matrix must be square with matching dimension to vertices")
    return vertices @ matrix.T


def translate_vertices(translation: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Translate a vertex set by ``t``."""
    if translation.ndim != 1:
        raise ValueError("translation must be 1D tensor")
    if vertices.ndim != 2:
        raise ValueError("vertices must be 2D tensor")
    if translation.size(0) != vertices.size(1):
        raise ValueError("translation and vertices dimension mismatch")
    return vertices + translation


def matmul_halfspace(
    matrix: torch.Tensor, normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply linear map ``x -> A x`` to an H-rep ``Bx <= c``."""
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
    """Translate an H-rep polytope by ``t``: offsets become ``c' = c + B t``."""
    if translation.ndim != 1:
        raise ValueError("translation must be a 1D tensor")
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, D) and offsets must be (F,)")
    if translation.size(0) != normals.size(1):
        raise ValueError("translation and normals dimensions mismatch")
    new_offsets = offsets + normals @ translation
    return normals.clone(), new_offsets


def lagrangian_product(
    vertices_P: torch.Tensor, vertices_Q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lagrangian product of two polytopes P, Q given their vertices.

    Returns ``(vertices, normals, offsets)`` in ``R^{2n}`` with block-diagonal H-rep.
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


# Random polytope generators ---------------------------------------------------
def random_polytope_algorithm1(
    seed: int | torch.Generator, num_facets: int, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random polytope via sampled halfspaces."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if num_facets < dimension + 1:
        raise ValueError("need at least d + 1 candidate halfspaces")
    generator = _make_generator(seed)
    dtype = torch.get_default_dtype()
    directions = _sample_in_unit_ball(generator, num_facets, dimension).to(dtype)
    normals = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
    offsets = torch.linalg.norm(directions, dim=1)
    eye = torch.eye(dimension, dtype=dtype)
    normals = torch.cat([normals, eye, -eye], dim=0)
    offsets = torch.cat(
        [
            offsets,
            torch.ones(dimension, dtype=dtype),
            torch.ones(dimension, dtype=dtype),
        ]
    )
    vertices = halfspaces_to_vertices(normals, offsets)
    cleaned_normals, cleaned_offsets = vertices_to_halfspaces(vertices)
    return vertices, cleaned_normals, cleaned_offsets


def random_polytope_algorithm2(
    seed: int | torch.Generator, num_vertices: int, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random polytope via sampled vertices."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if num_vertices < dimension + 1:
        raise ValueError("need at least d + 1 vertices")
    generator = _make_generator(seed)
    dtype = torch.get_default_dtype()
    raw_vertices = _sample_in_unit_ball(generator, num_vertices, dimension).to(dtype)
    centred = raw_vertices - raw_vertices.mean(dim=0, keepdim=True)
    normals, offsets = vertices_to_halfspaces(centred)
    vertices = halfspaces_to_vertices(normals, offsets)
    return vertices, normals, offsets
