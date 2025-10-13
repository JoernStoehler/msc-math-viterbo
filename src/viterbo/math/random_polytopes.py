"""Random polytope generators (stubs).

Algorithms to generate random convex polytopes, either via random halfspaces or
random vertices, with basic post-processing (e.g., redundancy removal).
"""

from __future__ import annotations

import torch

from .halfspaces import halfspaces_to_vertices, vertices_to_halfspaces


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
    # We deliberately skip clamping tiny/zero norms: the Gaussian hits the
    # origin with probability zero, and if the PRNG returns a degenerate draw
    # we are happy to let it surface loudly instead of papering over it.
    directions = points / norms
    radii = torch.rand((count, 1), generator=generator) ** (1.0 / dimension)
    return directions * radii


def random_polytope_algorithm1(
    seed: int | torch.Generator, num_facets: int, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random polytope via random halfspaces.

    Sketch:
      - Sample x ~ Uniform(B^d(1)); set normals = x / ||x|| and offsets = ||x||.
      - Remove redundant halfspaces; ensure closed polytope.

    Args:
      seed: int | torch.Generator.
      num_facets: target number of facets (not exact).
      dimension: ambient dimension ``d``.

    Returns:
      (vertices, normals, offsets):
        - vertices: (M, d)
        - normals: (F, d)
        - offsets: (F,)
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if num_facets < dimension + 1:
        raise ValueError("need at least d + 1 candidate halfspaces")
    generator = _make_generator(seed)
    dtype = torch.get_default_dtype()
    directions = _sample_in_unit_ball(generator, num_facets, dimension).to(dtype)
    normals = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
    offsets = torch.linalg.norm(directions, dim=1)
    # Add axis-aligned bounding halfspaces for robustness
    eye = torch.eye(dimension, dtype=dtype)
    normals = torch.cat([normals, eye, -eye], dim=0)
    offsets = torch.cat([
        offsets,
        torch.ones(dimension, dtype=dtype),
        torch.ones(dimension, dtype=dtype),
    ])
    vertices = halfspaces_to_vertices(normals, offsets)
    cleaned_normals, cleaned_offsets = vertices_to_halfspaces(vertices)
    return vertices, cleaned_normals, cleaned_offsets


def random_polytope_algorithm2(
    seed: int | torch.Generator, num_vertices: int, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a random polytope via random vertices.

    Sketch:
      - Sample x ~ Uniform(B^d(1)); centre by subtracting the mean.
      - Remove redundant vertices.

    Args:
      seed: int | torch.Generator.
      num_vertices: target number of vertices (not exact).
      dimension: ambient dimension ``d``.

    Returns:
      (vertices, normals, offsets):
        - vertices: (M, d)
        - normals: (F, d)
        - offsets: (F,)
    """
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

