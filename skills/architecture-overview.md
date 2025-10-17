---
name: architecture-overview
description: Layering and boundaries are maintained under Good Code Loop; keep this for quick reference.
last-updated: 2025-10-17
---

# Architecture Overview (Condensed)

For authoritative rules and review checklists, use `skills/good-code-loop.md`.

## Layering (Quick Reference)

1. Math (`src/viterbo/math/`): pure tensor utilities; no I/O or hidden state; accept caller devices, return tensors.
2. Datasets (`src/viterbo/datasets/`): adapters and `collate_fn`s; handle ragged data via lists or padding+mask.
3. Models (`src/viterbo/models/`): training/experiments; optional CUDA; no math-layer dependencies introduced here.
4. C++ (`src/viterbo/_cpp/`): hotspots via pybind11; provide Python fallbacks; guard imports.

See also `skills/data-collation.md` for batching, and `skills/performance-discipline.md` for extensions driven by measured hotspots.
