# Architecture Overview (Reference)

This document records decisions and rationale that go beyond the everyday overview in `AGENTS.md`.
It is intended for maintainers or agents working on architecture/conventions.

- Stack & devices
  - PyTorch-first; CPU baseline across CI and default runs.
  - GPU is optional and used only in `models/` where it improves throughput for conventional ML.
  - Math functions accept the caller’s device; no implicit device moves.

- Dtypes & numerics
  - Dtype expectations are documented per function/docstring (math often float64; ML often float32).
  - Avoid silent downcasts. Prefer numerically stable formulations and document trade-offs.

- Layering (no cycles)
  - `viterbo.math`: pure, stateless Torch tensor functions; no I/O; no dataclasses.
  - `viterbo.datasets`: thin adapters, schemas, and ragged collators; may cache or precompute.
  - `viterbo.models`: experiments and training loops; no core math logic.
  - `_cpp`: C++ extensions via `torch.utils.cpp_extension` (CPU baseline), with Python fallbacks.

- Ragged data patterns
  - Use Python lists of tensors or padded tensors + masks depending on the consumer.
  - Collate functions live in `viterbo.datasets` (e.g., `collate_list`, `collate_pad`).

- C++ extensions
  - Baseline is CPU-only C++ compiled via `torch.utils.cpp_extension.load` with pybind11 bindings.
  - Keep a safe Python fallback for each extension to preserve portability (CI, devcontainers).
  - Add CUDA only when a clear hotspot is demonstrated by profiling and justified by complexity.

- Testing & CI philosophy
  - Smoke-first: quick validators and selective benchmarks in PRs; deeper tiers are opt-in.
  - Incremental selection for fast feedback loops; CI uses CPU-only Torch wheels for speed.
  - Docs are built in CI so documentation drift is surfaced alongside code changes.

- Imports & public surface
  - Prefer explicit imports from concrete modules; do not aggregate or re-export APIs in `__init__.py`.
  - No `__all__` surfaces. Import paths should remain stable by direct module references.

- Tasks & parallelization
  - Author small, self-contained briefs under `docs/tasks/` (YAML front matter, clear scope, ACs).
  - Keep branches short-lived; ensure CI green before review; record any deviations.

