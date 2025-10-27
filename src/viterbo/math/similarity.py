"""Similarity metrics between convex polytopes (stubs).

Scope and intent
- Distance measures between convex polytopes with Torch-first contracts.
- APIs are stubbed while 4D experiments mature; shapes and semantics are
  documented to enable early callers and tests.

Planned functions
- ``hausdorff_distance(vertices_a, vertices_b)`` — symmetric Hausdorff distance
  between convex hulls of two vertex sets in ``R^d``.
- ``hausdorff_distance_under_symplectic_group(vertices_a, vertices_b)`` —
  Hausdorff distance up to symplectomorphisms in even dimensions (``d = 2n``).
- ``support_l2_distance(vertices_a, vertices_b, samples)`` — Monte Carlo L2
  distance between support functions via random directions.

Notes:
- Implementations will reuse helpers from ``viterbo.math.polytope`` (support
  queries) and deterministic sampling strategies. All functions will preserve
  dtype/device.

See Also:
- ``viterbo.math.polytope.support``, ``viterbo.math.polytope.support_argmax``
- ``viterbo.math.symplectic.symplectic_form`` for symplectic invariances
"""

from __future__ import annotations

import torch


def hausdorff_distance(vertices_a: torch.Tensor, vertices_b: torch.Tensor) -> torch.Tensor:
    """Return the symmetric Hausdorff distance between two convex polytopes (stub).

    Args:
      vertices_a: ``(Ma, d)`` vertex set for polytope A.
      vertices_b: ``(Mb, d)`` vertex set for polytope B.

    Returns:
      Scalar float tensor once implemented.
    """
    raise NotImplementedError


def hausdorff_distance_under_symplectic_group(
    vertices_a: torch.Tensor, vertices_b: torch.Tensor
) -> torch.Tensor:
    """Hausdorff distance up to symplectomorphisms in even dimensions (stub).

    Args:
      vertices_a: ``(Ma, 2n)`` vertex set for polytope A.
      vertices_b: ``(Mb, 2n)`` vertex set for polytope B.

    Returns:
      Scalar float tensor once implemented.
    """
    raise NotImplementedError


def support_l2_distance(
    vertices_a: torch.Tensor, vertices_b: torch.Tensor, samples: int
) -> torch.Tensor:
    """Approximate L2 distance between support functions using random directions (stub).

    Args:
      vertices_a: ``(Ma, d)`` vertex set for polytope A.
      vertices_b: ``(Mb, d)`` vertex set for polytope B.
      samples: number of random directions to sample (deterministic generator).

    Returns:
      Scalar float tensor once implemented.
    """
    raise NotImplementedError
