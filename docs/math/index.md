---
title: Math API Overview
---

# Math API (Overview)

This section provides high-level notes for the math layer. Detailed
documentation has been moved into module and function docstrings alongside the
code under `src/viterbo/math/` to keep it close to implementations.

- Conventions
  - Torch-first API; public functions preserve dtype/device and perform no I/O.
  - Shapes and invariants are documented in code docstrings.
  - Even-dimension policy for symplectic helpers (`d = 2n`).

Where to read docs now
- `viterbo.math.polytope` — per-body geometry (support, H/V conversions, queries).
- `viterbo.math.constructions` — builders and linear transforms.
- `viterbo.math.volume` — volume routines and planned backends.
- `viterbo.math.symplectic` — symplectic form and random symplectomorphisms.
- `viterbo.math.capacity_ehz.*` — EHZ capacities, cycles, spectra (4D focus).
- `viterbo.math.similarity` — distances between polytopes (stubs).
- `viterbo.math.polar` — polar body helpers and Mahler links (stubs).

Kept as separate specs
- [cstar_constant_spec](cstar_constant_spec.md) — derivation and certification
  details for the `C*(X)` budgets used by the 4D oriented-edge solver.
