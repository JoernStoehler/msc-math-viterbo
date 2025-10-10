---
status: draft
created: 2025-10-10
workflow: task
summary: Roadmap for implementing viterbo.modern fully and decommissioning legacy modules.
---

# Task Brief: Migrate to the viterbo.modern Architecture

## 1. Objective

Deliver a production-ready implementation of the `viterbo.modern` namespace, exercise it end to end, and retire the legacy code paths without breaking existing workflows.

## 2. Current Context

- `viterbo.modern` currently consists of stub modules, placeholder tests, and illustrative notebooks.
- Legacy functionality still lives under `src/viterbo/` and its consumers/tests/notebooks.
- CI expects smoke-level pytest coverage, Ruff linting, and Pyright type checking.

## 3. Work Plan

1. **Map the surfaces**
   - Review the stub signatures, docstrings, and type conventions in `viterbo.modern` to confirm alignment with the architecture principles in `AGENTS.md`.
   - Catalogue any missing primitives needed for feature parity with the legacy modules.
2. **Replace stubs with implementations**
   - Implement pure JAX-first math routines (polytopes, generators, capacities, spectra, cycles) with explicit padding strategies.
   - Fill in dataset/converter helpers to cover Polars integration, schema enforcement, and padding conversions.
   - Keep notebooks executable by updating them to call the new surfaces.
3. **Design validation**
   - Draft a test matrix covering unit correctness, geometric invariants, numerical stability, and batching behaviour.
   - Add/expand pytest suites with proper goal/suite markers; include property-based tests where useful.
   - Plan performance benchmarks if any new algorithms are asymptotically heavier than the legacy versions.
4. **Quality gates**
   - Ensure Ruff, Pyright, and `pytest` (smoke tier) pass locally and in CI; add deep/longhaul runs if risk dictates.
   - Execute the notebooks end to end (script mode) to confirm dataset build/consume flows succeed.
5. **Review cycle**
   - Conduct an internal review of API ergonomics, padding semantics, and documentation before requesting maintainer review.
   - Address feedback, update docstrings/tutorial notebooks, and confirm all automation stays green.
6. **Migration and cleanup**
   - Remove deprecated legacy modules, tests, and notebooks once the modern replacements are verified.
   - Update docs, briefs, and any orchestration scripts to point to `viterbo.modern` as the canonical entry point.
   - Record migration notes or ADRs if major architectural decisions change.

## 4. Deliverables

- Fully implemented `viterbo.modern` package with comprehensive tests and benchmarks.
- Updated notebooks illustrating atlas generation and consumption using the new APIs.
- CI evidence (lint, typecheck, pytest) demonstrating readiness.
- Migration commits removing obsolete code and updating documentation/briefs.

## 5. Dependencies & Open Questions

- Confirm data storage conventions (Arrow/Parquet formats) once dataset helpers solidify.
- Coordinate with maintainers on timing for removing legacy entry points.
- Determine whether additional tooling (e.g., benchmarking harnesses) is required before the cleanup phase.

## 6. Success Criteria

- All legacy consumers can switch to `viterbo.modern` without regressions in coverage, performance, or maintainability.
- CI remains green post-migration, and notebooks execute without manual intervention.
- Documentation reflects the new architecture, and no deprecated references remain in the repo.
