r"""Polar duals and Mahler link stubs.

This module will house polar body computations and helpers connecting symplectic
inequalities to Mahler-type products via polarity (e.g., K × K^\circ links).
All functions are pure and torch-first.

See Also:
- ``viterbo.math.volume.volume`` for Mahler-type products once implemented
- ``viterbo.math.constructions`` for canonical shapes to test polarity
"""

from __future__ import annotations

import torch


def polar_from_halfspaces(
    normals: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the polar body ``K^\circ`` given an H-rep of ``K`` (stub).

    The polar is defined by ``K^\circ = \{ y \mid \langle y, x \rangle \le 1\ \forall x \in K \}``.
    The stable implementation will normalise the H-rep and compute either an
    explicit V-representation for K^\circ or another H-representation depending
    on caller needs.

    Args:
      normals: ``(F, d)`` float tensor of outward normals for K.
      offsets: ``(F,)`` float tensor of offsets with positive entries.

    Returns:
      (normals_dual, offsets_dual) describing K^\circ in H-representation once implemented.
    """
    if normals.ndim != 2 or offsets.ndim != 1:
        raise ValueError("normals must be (F, d) and offsets must be (F,)")
    raise NotImplementedError


def mahler_product_approx(vertices: torch.Tensor) -> torch.Tensor:
    r"""Approximate Mahler product ``vol(K) * vol(K^\circ)`` (stub).

    Will rely on volume backends and a robust polar routine; initially intended
    for low-dimensional certification and 4D exploration.

    Args:
      vertices: ``(M, d)`` float tensor of vertices of K.

    Returns:
      Scalar float tensor once backends are available.
    """
    if vertices.ndim != 2:
        raise ValueError("vertices must be (M, d)")
    raise NotImplementedError
