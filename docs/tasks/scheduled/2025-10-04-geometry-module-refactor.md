# Task Brief — Geometry quantity module restructure and JAX baselines

- **Status**: Scheduled
- **Last updated**: 2025-10-04
- **Owner / DRI**: Unassigned
- **Related docs**: `docs/tasks/01-task-evaluation-methodology.md`, `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`, `docs/22-code-structure.md`

## 1. Context and intent
The geometry package currently mixes reference algorithms, optimisation shortcuts, and experimental implementations inside shared modules. This makes it hard to compare algorithm families, build trustworthy datasets, or introduce JAX for higher-quality numerics. The maintainer has requested a per-quantity folder layout containing reference-first implementations alongside optional optimised or JAX variants. This task delivers that restructure so downstream experiments (facet dataset, MILP relaxations, correlation studies) can assume consistent module surfaces.

## 2. Objectives and non-goals

### In scope
- Introduce per-quantity packages (e.g., `viterbo.geometry.volume`, `viterbo.geometry.polytopes`) each with `__init__.py`, helper modules, fixtures, and example data.
- Provide reference implementations (`*_reference.py`) that favour readability and alignment with the mathematical definitions.
- Add optional `*_optimized.py` and `*_jax.py` siblings when performance or differentiability is beneficial; ensure signatures remain comparable.
- Seed each quantity package with small curated example datasets and documentation on how to select implementations.
- Draft initial JAX coverage targeting linear-algebra kernels that unblock dataset generation for Experiment E1.
- Document migration notes for contributors (e.g., import paths, helper relocation) and update public APIs accordingly.

### Out of scope
- Wholesale rewrites of algorithms beyond what is needed to factor shared helpers.
- High-performance GPU kernels or distributed execution.
- Adding new mathematical quantities beyond those already listed in the algorithm implementation plan.

## 3. Deliverables and exit criteria
- Updated directory layout under `src/viterbo/geometry/` reflecting the quantity-first structure with clear module docstrings.
- Reference, optimised, and (where applicable) JAX implementations exposing harmonised signatures and documented behaviour differences.
- Regression smoke tests confirming parity between implementation variants on shared fixtures.
- Example datasets stored under each quantity package (e.g., simple polytopes) with loaders or helper functions.
- Contributor documentation summarising how to pick between variants, including JAX usage notes and any temporary blockers.

## 4. Dependencies and prerequisites
- Inventory of current geometry algorithms and consumers (`rg` scan + reading existing modules).
- Alignment with AGENTS.md conventions (already satisfied by Task 001).
- No external dependencies beyond NumPy/SciPy/JAX; confirm JAX is already available in the environment or raise an unblocker.

## 5. Execution plan and checkpoints
1. **Survey (Day 0–1)**: Catalogue existing geometry algorithms, identify candidate quantity groupings, and map implicit helpers.
2. **Layout proposal (Day 1)**: Sketch the target directory tree and verify with the maintainer if any naming disputes appear.
3. **Reference extraction (Day 2–4)**: Move reference implementations into their packages, add module docstrings, and wire shared helpers.
4. **Variant alignment (Day 3–5)**: Adapt optimised implementations to the new locations; ensure parity tests cover signatures and invariants.
5. **Initial JAX hooks (Day 4–6)**: Port linear-algebra kernels (`standard_symplectic_matrix`, dense solves, antisymmetric products) into JAX modules; wrap deterministic enumeration utilities with JAX-friendly APIs.
6. **Fixtures and docs (Day 5–6)**: Populate example datasets, update documentation, and write contributor notes.
7. **Final review (Day 7)**: Run `make ci`, ensure benchmarks still point to valid paths, and request maintainer review with a migration summary.

## 6. Effort and resource estimates
- **Agent time**: Medium (≈ 1.5 agent-weeks)
- **Compute budget**: Low (CPU unit tests and small benchmarks)
- **Expert/PI involvement**: Low (spot-check layout and confirm mathematical invariants if questions arise)

## 7. Testing, benchmarks, and verification
- CI: `make format`, `make lint`, `make typecheck`, and targeted unit suites covering moved modules.
- Add or update smoke tests comparing reference vs. optimised vs. JAX variants on shared fixtures.
- Optional local run of the medium benchmark tier (<20 minutes) to confirm no catastrophic regressions.

## 8. Risks, mitigations, and escalation triggers
- **Risk**: Hidden coupling between geometry helpers and consumers causes breakage. **Mitigation**: Stage refactor with deprecation adapters and run the full unit suite after each quantity move.
- **Risk**: JAX parity harder than expected for SciPy-backed routines. **Mitigation**: Document blockers explicitly, keep reference path canonical, and defer advanced kernels to follow-up tasks.
- **Escalation triggers**: Discovery that a required algorithm lacks tests, missing JAX dependency support, or inability to harmonise signatures without breaking public APIs.

## 9. Follow-on work
- Task for benchmark and profiling harness (T2) to leverage the new layout.
- Experiment E1 (facet-normal dataset) consuming the refactored modules.
- Future tasks for extending JAX coverage once blockers are resolved (e.g., SciPy replacements).
