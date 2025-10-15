"""Symplectic linear algebra utilities.

This module exposes the standard symplectic form and a random symplectic
matrix generator. Constructions live in ``viterbo.math.constructions`` and
capacity/cycle solvers live in ``viterbo.math.capacity_ehz.*``.
"""

from __future__ import annotations

import torch


def symplectic_form(dimension: int) -> torch.Tensor:
    """Return the standard symplectic form ``J`` of size ``(d, d)``."""
    if dimension <= 0 or dimension % 2 != 0:
        raise ValueError("dimension must be a positive even integer")
    half = dimension // 2
    dtype = torch.get_default_dtype()
    eye = torch.eye(half, dtype=dtype)
    top = torch.cat([torch.zeros_like(eye), eye], dim=1)
    bottom = torch.cat([-eye, torch.zeros_like(eye)], dim=1)
    return torch.cat([top, bottom], dim=0)


def random_symplectic_matrix(dimension: int, seed: int | torch.Generator) -> torch.Tensor:
    """Return a random symplectic matrix ``M`` satisfying ``M.T @ J @ M = J``."""
    if dimension <= 0 or dimension % 2 != 0:
        raise ValueError("dimension must be a positive even integer")
    generator = torch.Generator(device="cpu")
    if isinstance(seed, torch.Generator):
        generator = seed
    else:
        generator.manual_seed(int(seed))
    dtype = torch.get_default_dtype()
    half = dimension // 2
    random_matrix = torch.randn((half, half), generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(random_matrix)
    a = q
    sym_upper = torch.randn((half, half), generator=generator, dtype=dtype)
    sym_upper = (sym_upper + sym_upper.T) / 2.0
    sym_lower = torch.randn((half, half), generator=generator, dtype=dtype)
    sym_lower = (sym_lower + sym_lower.T) / 2.0

    identity = torch.eye(half, dtype=dtype)
    block_a = torch.block_diag(a, torch.linalg.inv(a.T))
    upper = torch.block_diag(identity, identity)
    upper = upper.clone()
    upper[:half, half:] = sym_upper
    lower = torch.block_diag(identity, identity)
    lower = lower.clone()
    lower[half:, :half] = sym_lower
    matrix = upper @ block_a @ lower
    return matrix
