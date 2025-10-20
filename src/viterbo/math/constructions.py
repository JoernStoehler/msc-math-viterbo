"""Geometric constructions of simple polytopes and configurations.

Torch-first helpers that return vertex and/or halfspace representations.
"""

from __future__ import annotations

import math

import torch

from viterbo.math.polytope import halfspaces_to_vertices, vertices_to_halfspaces
from viterbo.math.volume import volume as polytope_volume


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


# -----------------------------------------------------------------------------
# Canonical, deterministic constructors (Torch-first, no I/O)


def unit_square() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Axis-aligned square [-1, 1]^2 in R^2.

    Returns (vertices, normals, offsets) with float64 CPU tensors.
    - Vertices are ordered counter-clockwise.
    - Deterministic with no randomness.
    - Normals are unit-length; offsets positive.
    - Area (via viterbo.math.volume.volume) equals 4.0.
    """
    dtype = torch.float64
    device = torch.device("cpu")
    vertices = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=dtype, device=device
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    # Area sanity (not used, but documents determinism in implementation).
    _ = polytope_volume(vertices)
    return vertices, normals, offsets


def triangle_area_one() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right triangle with unit area in R^2.

    Uses vertices [[0,0], [2,0], [0,1]] in CCW order. Returns float64 CPU tensors.
    Deterministic; normals are unit-length and offsets positive.
    """
    dtype = torch.float64
    device = torch.device("cpu")
    vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=dtype, device=device)
    normals, offsets = vertices_to_halfspaces(vertices)
    return vertices, normals, offsets


def regular_simplex(d: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonal simplex in R^d: origin + standard basis vectors.

    Args:
      d: dimension (d >= 1).

    Returns:
      (vertices, normals, offsets) as float64 CPU tensors.

    Notes:
      For d = 4, the volume equals 1/4!.
    """
    if d < 1:
        raise ValueError("dimension d must be >= 1")
    dtype = torch.float64
    device = torch.device("cpu")
    origin = torch.zeros((1, d), dtype=dtype, device=device)
    basis = torch.eye(d, dtype=dtype, device=device)
    vertices = torch.cat((origin, basis), dim=0)
    normals, offsets = vertices_to_halfspaces(vertices)
    return vertices, normals, offsets


def counterexample_pentagon_product() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regular 5-gon × 90°-rotated 5-gon in R^4.

    Uses rotated_regular_ngon2d(5, 0.0) and rotated_regular_ngon2d(5, -0.5*pi),
    then combines factors via lagrangian_product(). Returns float64 CPU tensors.
    """
    v_q, _, _ = rotated_regular_ngon2d(5, 0.0)
    v_p, _, _ = rotated_regular_ngon2d(5, -0.5 * torch.pi)
    vertices, normals, offsets = lagrangian_product(v_q.to(torch.float64), v_p.to(torch.float64))
    return vertices, normals, offsets


def noisy_pentagon_product(
    seed_q: int = 314159, seed_p: int = 271828, amp: float = 0.03
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noisy 5-gon × rotated 5-gon product in R^4 (deterministic).

    Applies slight per-vertex radial noise to each factor before forming the
    Lagrangian product. Deterministic given seeds; returns float64 CPU tensors.

    Args:
      seed_q: RNG seed for q-plane 5-gon.
      seed_p: RNG seed for p-plane 5-gon.
      amp: amplitude of per-vertex radial noise (scale = 1 + amp*(U-0.5)).
    """
    if amp < 0:
        raise ValueError("amp must be non-negative")
    gen_q = torch.Generator(device="cpu")
    gen_q.manual_seed(int(seed_q))
    gen_p = torch.Generator(device="cpu")
    gen_p.manual_seed(int(seed_p))
    base_q, _, _ = rotated_regular_ngon2d(5, 0.0)
    base_p, _, _ = rotated_regular_ngon2d(5, -0.5 * torch.pi)
    scale_q = 1.0 + amp * (torch.rand(5, generator=gen_q) - 0.5)
    scale_p = 1.0 + amp * (torch.rand(5, generator=gen_p) - 0.5)
    vertices_q = (base_q * scale_q.unsqueeze(1)).to(torch.float64)
    vertices_p = (base_p * scale_p.unsqueeze(1)).to(torch.float64)
    vertices, normals, offsets = lagrangian_product(vertices_q, vertices_p)
    return vertices, normals, offsets


def mixed_nonproduct_from_product(
    mix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a fixed near-identity mixing to a noisy pentagon product in R^4.

    Starts from noisy_pentagon_product() and applies a linear mixing matrix to
    break block-diagonal (product) structure, then recomputes H-representation.

    Args:
      mix: optional (4,4) matrix. If provided, dtype/device are preserved.
           Defaults to the matrix used in tests/polytopes.py.
    """
    base_vertices, _, _ = noisy_pentagon_product()
    if mix is None:
        mix = torch.tensor(
            [
                [1.0, 0.02, 0.01, 0.0],
                [0.0, 0.99, -0.03, 0.02],
                [0.01, 0.0, 1.0, 0.04],
                [0.0, -0.02, 0.0, 1.02],
            ],
            dtype=torch.float64,
            device=base_vertices.device,
        )
    if mix.ndim != 2 or mix.size(0) != 4 or mix.size(1) != 4:
        raise ValueError("mix must be a (4, 4) matrix")
    vertices = base_vertices.to(dtype=mix.dtype, device=mix.device) @ mix.T
    normals, offsets = vertices_to_halfspaces(vertices)
    return vertices, normals, offsets


def random_polygon(seed: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic planar polygon via angle-sorted radii in [0.8, 1.2].

    Args:
      seed: RNG seed (CPU generator).
      k: number of vertices (k >= 3).

    Returns:
      (vertices, normals, offsets) with float64 CPU tensors. Vertices are in
      CCW order (increasing angle).
    """
    if k < 3:
        raise ValueError("k must be at least 3")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    raw = torch.rand(k, generator=gen) * (2.0 * torch.pi)
    # Map to [-pi, pi) then sort so atan2 order is non-decreasing (CCW).
    mapped = raw.clone()
    mapped = torch.where(mapped > torch.pi, mapped - 2.0 * torch.pi, mapped)
    order = torch.argsort(mapped)
    angles = mapped[order]
    radii = 0.8 + 0.4 * torch.rand(k, generator=gen)
    radii = radii[order]
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    vertices = torch.stack((x, y), dim=1).to(dtype=torch.float64, device=torch.device("cpu"))
    normals, offsets = vertices_to_halfspaces(vertices)
    return vertices, normals, offsets


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
