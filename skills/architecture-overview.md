---
name: architecture-overview
description: This skill should be used when understanding layer responsibilities, data flow, and extension points across math, datasets, models, and C++ bindings.
last-updated: 2025-10-17
---

# Architecture Overview

## Instructions
- Validate intended changes against layer responsibilities below; never import upward (`math` ← `datasets` ← `models`).
- Keep math-layer APIs pure and device-agnostic; move side effects to adapters or call sites.
- For performance work, start in Python/Torch and escalate before adding C++ or CUDA.
- If a change crosses layers or adds new dependencies, flag the task with `Needs-Unblock: architecture`.

## Layering

1. **Math (`src/viterbo/math/`)**
   - Pure tensor utilities; no I/O or hidden state.
   - Accept caller devices, return tensors, and maintain deterministic behavior.
2. **Datasets (`src/viterbo/datasets/`)**
   - Adapters and collate functions that wrap math utilities.
   - Handle ragged data via Python lists of tensors or padded tensors with masks.
3. **Models (`src/viterbo/models/`)**
   - Training loops and experiment orchestration.
   - May use CUDA but must not introduce math-layer dependencies.
4. **C++ Extensions (`src/viterbo/_cpp/`)**
   - Performance hotspots implemented via pybind11.
   - Provide Python fallbacks where feasible; coordinate with maintainers before adding new bindings.

## Data Handling

- Ragged inputs: prefer explicit masks or lists; expose `collate_fn`s for DataLoaders.
- Device management: no implicit `.to()` calls; leave device decisions to callers.
- Precision: math defaults to float64, models to float32. Document any deviations.

## Extension Strategy

- Start with pure Python/Torch; introduce C++ only for demonstrable hotspots.
- When adding extensions, document build requirements and guard imports to degrade gracefully.

## Related Skills

- `math-layer` — detailed guidance for geometry modules.
- `coding-standards` — cross-cutting style and purity rules.
- `performance-discipline` — governs profiling and regression handling.
