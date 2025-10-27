"""Derived ratios and scalars built from capacities.

Systolic ratio normalisation
- We adopt the mainstream literature normalisation
  ``Sys(K^{2n}) = c_{EHZ}(K)^{n} / (n! Vol_{2n}(K))``.

References (definitions and usage)
- C. Viterbo, *Functors and computations in Floer homology with applications*,
  GAFA 9 (1999), and subsequent expositions of the volume–capacity conjecture.
- S. Artstein-Avidan, R. Karasev, Y. Ostrover, *From Symplectic Measurements to
  the Mahler Conjecture*, Duke Math. J. 163 (2014). See §1 for the
  normalisation and its relation to isoperimetric-type inequalities.
- P. Haim‑Kislev, Y. Ostrover, *A Counterexample to Viterbo’s Conjecture*,
  arXiv:2405.16513 (2024). Uses the same normalisation when reporting
  violations in dimension 4.

See Also:
- ``viterbo.math.volume.volume`` for computing ``Vol_{2n}(K)``
- ``viterbo.math.capacity_ehz.algorithms`` for ``c_{EHZ}(K)``
"""

from __future__ import annotations

import math

import torch


def systolic_ratio(
    volume: torch.Tensor, capacity_ehz: torch.Tensor, symplectic_dimension: int | None = None
) -> torch.Tensor:
    """Systolic ratio ``c_{EHZ}(K)^{n} / (n! Vol_{2n}(K))`` for ``dim K = 2n``.

    Notes:
    - This matches the convention where the Euclidean ball in ``R^{2n}``
      satisfies ``Sys(B^{2n}) = 1``.
    - Previous revisions in this codebase used the reciprocal normalisation
      ``Vol/c^{n}``; the current definition conforms with standard
      references cited above.
    """
    if volume.ndim != 0 or capacity_ehz.ndim != 0:
        raise ValueError("volume and capacity_ehz must be scalar tensors")
    if torch.any(capacity_ehz <= 0):
        raise ValueError("capacity_ehz must be strictly positive")
    if symplectic_dimension is None:
        raise ValueError("symplectic_dimension must be provided for systolic_ratio")
    if symplectic_dimension % 2 != 0 or symplectic_dimension <= 0:
        raise ValueError("symplectic_dimension must be a positive even integer")
    n = symplectic_dimension // 2
    denom = (
        torch.tensor(float(math.factorial(n)), dtype=volume.dtype, device=volume.device) * volume
    )
    return capacity_ehz.pow(n) / denom
