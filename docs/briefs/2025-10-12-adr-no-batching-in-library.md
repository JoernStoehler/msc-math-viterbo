---
status: adopted
created: 2025-10-12
workflow: adr
summary: Library APIs are per-instance only; batching/padding lives in ML pipelines.
---

# ADR: No Batching in Library APIs

## Decision

- Core library APIs operate on a single polytope (or polytope pair) per call.
- Remove batching/padding concerns from library kernels; no `NaN` padding or mask conventions.
- ML/training code is responsible for batching and padding.

## Rationale

- The atlas is computed once (row-by-row); simplicity beats peak throughput.
- Batching introduces shape/padding semantics that complicate testing and maintenance.
- Downstream ML defines its own batching and padding policies anyway.

## Transition Plan

- Mark existing batched entry points as deprecated in docs; avoid adding new ones.
- Keep current batched functions temporarily for compatibility with existing tests and examples.
- New code and briefs should use per-instance APIs only; remove batched variants after the next two milestones.

## Guidance

- Write per-row compute adapters over the HF Datasets atlas (`map`-style transforms).
- Return native Python scalars or lists (converted from JAX types) where appropriate for dataset storage.
- Keep JAX purity in math code; perform I/O and dataset interactions in thin adapters.

