---
name: data-collation
description: Use for batching ragged data, designing collate functions, and keeping math pure.
last-updated: 2025-10-17
---

# Data & Collation

## Instructions
- Keep collation in datasets/models; math remains batching-agnostic and side-effect free.
- Choose between lists of tensors and padded tensors with masks based on downstream ops; document the contract.
- Provide `collate_fn` helpers for DataLoaders and test representative ragged cases.

## Policies

- Prefer explicit masks when padding; ensure shapes and semantics are documented (batch, length, dims).
- Accept Python lists of tensors where padding is wasteful or distorts semantics.
- Keep conversions (e.g., `.detach().cpu().numpy()`) at call sites; library APIs stay tensor-native.

## Dataloader Integration

- Surface a `collate_fn` that:
  - Validates input shapes and optional masks.
  - Assembles batches without implicit device moves or dtype casts.
  - Plays well with `num_workers>0` (avoid capturing large globals).

## Testing

- Add smoke tests that exercise ragged batches and masks; assert invariants and shape agreements.
- Seed randomness with `torch.Generator` where stochastic batching is involved.

## Performance Notes

- Padding can improve vectorization but may increase memory; measure with representative inputs.
- For performance regressions or CPU hotspots, see `performance-discipline`.

## Related Skills

- `good-code-loop` — broader coding and validation loop.
- `math-layer` — math APIs should remain batching-agnostic.
- `performance-discipline` — when batching choices impact throughput.

