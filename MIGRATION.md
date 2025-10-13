# PyTorch + C++ Migration (MIGRATION.md)

This document tracks goals, decisions, scope, milestones, and progress for migrating the project from a JAX‑first stack to a PyTorch + C++ stack. It serves as the single source of truth during migration.

## Goals

- Replace JAX‑first math layer with PyTorch (CPU/GPU) and a path for C++ kernels where control flow or ragged structures dominate.
- Simplify typing: use Pyright with clear docstrings and lightweight shape/dtype notes; drop `jaxtyping` from public APIs.
- Keep repo fast to iterate: small, readable pure‑Python math first; add C++ only on hot paths.
- Preserve three main modules with clear responsibilities:
  - `viterbo.math`: pure geometry/math utilities (PyTorch tensors in/out; deterministic; no I/O).
  - `viterbo.datasets`: orchestration to produce Torch datasets, collate functions for ragged data.
  - `viterbo.models`: experiments to probe Viterbo conjecture; training/eval scaffolds.
- Retain DevOps scripts and domain notebooks. Reduce the rest to minimal, working examples.

## Scope and Non‑Goals (initial)

- In scope: repo cleanup; dependency/tooling switch; scaffolding minimal math/datasets/models; CI smoke; minimal examples/tests; staged deletion of JAX code.
- Out of scope (initial): re‑implementing all algorithms; advanced performance work; HF datasets; rich docs site; longhaul benches.

## Early Architecture Decisions (proposed)

- Framework: PyTorch >= 2.4 (stable), Python 3.12 baseline.
- Device policy: CPU baseline; optional CUDA if available (`torch.cuda.is_available()` guards). No device‑specific logic in math functions—accept tensor device from caller.
- Ragged data: represent as Python lists of tensors or padded tensors with masks; expose `collate_fn` options (`pad_left`, `pad_right`, `stack_list`) under `datasets`.
- C++ interop: use `torch.utils.cpp_extension` with pybind11 shims; CMake optional; start with a header‑only example and build harness (no CUDA required initially).
- Reproducibility: prefer passing `torch.Generator`/seed integers explicitly; avoid hidden global state in math.
- Logging: standard `logging` in non‑hot paths; avoid logging inside JIT/C++ kernels.

## Tooling & DevOps

- Dependency manager: keep `uv`; update `pyproject.toml` and `uv.lock` (drop JAX/equinox/jaxtyping; add `torch`).
- Linters/typecheckers: keep Ruff and Pyright; remove jaxtyping‑specific ignores; keep pragmatic docstring policy.
- CI: preserve smoke tier with coverage; temporarily skip deep/longhaul until critical paths are ported.
- Justfile: keep golden‑path commands; update `lint`, `typecheck`, `test`, and add `build-cpp` sample.

## Deletion & Refactor Strategy

1) Archive JAX era: tag the last JAX commit (e.g., `v0-jax`) for future reference.
2) Stage deletions: remove most `src/viterbo/**` content except the new minimal skeleton; keep notebooks, scripts, docs shell.
3) Introduce new structure under `src/viterbo/`:
   - `math/`: a few pure geometry helpers using Torch.
   - `datasets/`: a minimal `torch.utils.data.Dataset` and `collate_fn` for ragged data.
   - `models/`: tiny model and training loop example.
   - `_cpp/`: placeholder for C++/pybind11 sources and a build recipe.
4) Replace tests with small smoke tests for imports and a single end‑to‑end sample.

## Milestones & Acceptance Criteria

- M0 – Planning (this doc)
  - Goals, decisions, scope, and plan written; open questions recorded.

- M1 – Skeleton + Tooling switch
  - `pyproject.toml` updated (Torch added; JAX removed); `uv.lock` refreshed.
  - New `src/viterbo/{math,datasets,models}` scaffold compiles; imports succeed.
  - CI smoke green; minimal tests in place.
  - CI installs CPU-only torch (`just ci-cpu`); local dev may use GPU wheels.

- M2 – Core math utilities
  - Basic geometry routines (e.g., norms, support, simple polytope helpers) ported with tests.
  - Ragged collate functions implemented and covered by tests.

- M3 – C++ extension scaffold
  - Example C++ op built via `torch.utils.cpp_extension`; Python bindings tested on CI (CPU only).

- M4 – Cleanup & docs
  - Old JAX modules/tests removed; README/docs updated to new stack; brief migration notes published.

## Progress Log

- 2025‑10‑13: Drafted migration plan and proposed decisions; awaiting confirmation on open questions.
- 2025‑10‑13: Skeletonized repo to PyTorch: updated AGENTS.md, pyproject/pytest/Justfile; added minimal `math`/`datasets`/`models`; added C++ extension scaffold; smoke + benchmark tests; added dummy notebooks for artefacts I/O.

## Open Questions (need PI confirmation)

1. Torch version and accelerators
   - OK to standardize on PyTorch 2.4 (stable) and CPU baseline? Enable CUDA optionally only when present (no hard dependency)?
2. C++ route
   - Prefer `torch.utils.cpp_extension` + pybind11 without CMake initially? CUDA deferred?
3. Deletions cadence
   - Proceed with immediate skeletonization (delete most JAX code/tests now) or keep a short compatibility window with a thin legacy namespace before full cutover?
4. Notebooks and docs
   - Keep `notebooks/` as‑is and prune `docs/` to a minimal readme + this MIGRATION.md for now?
5. CI depth
   - During migration, is smoke‑only sufficient on PRs, with deep/bench scheduled later?
6. Torch CPU vs GPU install policy
   - Adopt CPU-only wheels in CI via `just ci-cpu` (uv pip with CPU index), keep lock lean; allow GPU wheels locally.

## Next Actions (pending confirmation)

1. Update `pyproject.toml`: remove JAX stack; add `torch`; keep dev deps.
2. Scaffold minimal `math`, `datasets`, `models` with simple examples and smoke tests.
3. Prune tests to smoke tier; disable/skip legacy suites.
4. Keep `Justfile` but adjust tasks for Torch/C++.
5. Stage deletions of legacy modules; keep notebooks and scripts.

## Risks & Mitigations

- Binary build complexity (C++/CUDA): start CPU‑only; add CUDA when required; use GitHub Action matrix later.
- Performance regressions: accept initially; add targeted C++ kernels after profiling.
- API churn: keep functions small and semantic; document with concise docstrings.
