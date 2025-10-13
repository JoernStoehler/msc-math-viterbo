"""Random polytope generators (stubs).

Algorithms to generate random convex polytopes, either via random halfspaces or
random vertices, with basic post-processing (e.g., redundancy removal).
"""

from __future__ import annotations

import torch


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
    raise NotImplementedError


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
    raise NotImplementedError

