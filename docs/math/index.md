---
title: Math API Overview
---

# Math API (Overview)

This section documents the mathematical surface exposed under `viterbo.math`.
It focuses on semantics, invariants, shapes, and dtypes; implementation details
are intentionally omitted. All functions are pure (no I/O, no hidden state) and
accept tensors on the caller’s device without implicit moves.

- Conventions
  - Torch-first API; return tensors from public functions.
  - Documented shapes and dtypes per function; avoid silent downcasts.
  - Even-dimension policy for symplectic helpers (`d = 2n`).

Modules

- [polytope](polytope.md) — per-body geometry (support, H/V conversions, queries).
- [constructions](constructions.md) — builders and linear transforms.
- [volume](volume.md) — volume routines and future backends.
- [symplectic](symplectic.md) — symplectic form and random symplectomorphisms.
- [minimal_action](minimal_action.md) — EHZ capacities, cycles, spectra (4D focus).
- [similarity](similarity.md) — distances between polytopes (stubs).
- [polar](polar.md) — polar body helpers and Mahler links (stubs).
