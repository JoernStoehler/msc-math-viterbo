---
title: symplectic — linear algebra basics
---

# `viterbo.math.symplectic`

Linear-algebra helpers for symplectic geometry. Ambient dimension must be even
(`d = 2n`).

Functions

- `symplectic_form(dimension: int) -> (d, d)`
  - Returns the standard symplectic form `J = [[0, I], [-I, 0]]`.

- `random_symplectic_matrix(dimension: int, seed: int|Generator) -> (d, d)`
  - Deterministically generates a symplectic matrix satisfying `M^T J M = J`
    using QR-based factorisations and symmetric shear blocks.

Notes

- Use `viterbo.math.constructions.lagrangian_product` for polytope products and
  `viterbo.math.capacity_ehz.*` for capacity/cycle computation.
