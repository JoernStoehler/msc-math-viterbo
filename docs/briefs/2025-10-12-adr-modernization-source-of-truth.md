---
status: adopted
created: 2025-10-12
workflow: adr
summary: Single source of truth for the modern codebase scope and priorities.
---

# ADR: Modernization — Source of Truth

## Decision

- Adopt the flat `viterbo` namespace as the production surface. The modern stack
  includes capacity (facet-solvers, Reeb graph, Minkowski billiards), spectrum (4D),
  geometry helpers, and a central dataset (“atlas”).
- Focus on 4D first; ≥6D work is deferred until the next milestone.
- Prefer Hugging Face Datasets for the atlas backbone (see the companion ADR).
- Library APIs are per-instance (no batching/padding at the library layer). ML code
  handles batching as needed (see the companion ADR).
- Retire legacy planning briefs; maintain this ADR plus focused task briefs only.

## Rationale

The project operates at modest dataset scale (~1e5 rows) and prioritises clarity and
reproducibility over throughput in core kernels. HF Datasets offers ergonomic row-first
operations, built-in Parquet support, and straightforward cloud storage. Batching belongs
in model training pipelines, not in geometry/capacity kernels.

## Scope & Boundaries (today)

- Quantities
  - EHZ capacity: reference and fast facet-normal solvers; Reeb-cycle wrappers; Minkowski
    billiard length for products and diagnostics.
  - Spectrum: 4D oriented-edge baseline with deterministic enumeration and ordering.
  - Volumes: robust reference estimators.
- Datasets
  - Atlas built on HF Datasets with Parquet backing; rows as dicts; deterministic seeds and
    provenance fields.
- Out of scope
  - ≥6D capacity/spectrum/cycles until combinatorial scaling and CI budget are defined.
  - Library-level batching and padding.

## Migration Notes

- Older briefs under `docs/briefs/` have been retired. This ADR supersedes the previous
  “modernization roadmap” and related migration notes. The modern `viterbo.symplectic` module
  remains part of the flat namespace; only the legacy package tree was removed.

## Follow-ups

1. Implement HF Datasets atlas adapters and smoke tests.
2. Mark legacy batched APIs as deprecated in docs; keep compatibility while tests transition.
3. Track ≥6D planning separately when ready (runtime envelopes, combinatorics, fixtures).

