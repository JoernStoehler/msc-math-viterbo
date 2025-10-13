---
status: proposed
created: 2025-10-13
workflow: task
summary: Restructure viterbo.datasets2 around explicit helpers, example builders, and updated docs/benchmarks.
---

# Task: Streamlined datasets helper stack

## Goals

1. Replace the remaining scaffolding in `viterbo.datasets2` with explicit helper modules for schemas, converters, and quantity evaluation.
2. Provide two clear dataset builders (`atlas_tiny`, `atlas_small`) that regenerate from scratch and illustrate full vs. production algorithm coverage.
3. Update documentation, smoke tests, and benchmarks to reflect the new helpers and ensure performance baselines are recorded.

## Deliverables

1. `schema.py` with reusable column Feature definitions and notes on non-persisted intermediates (e.g., incidence matrices).
2. `converters.py` housing row⇄JAX conversion helpers for polytopes, cycles, and numeric columns compatible with Hugging Face datasets.
3. `quantities.py` exposing algorithm-selecting wrappers (typed `Literal[...]` switches) that validate required columns, call `viterbo.math`, and normalise return types (e.g., floats, `(value, cycle)` tuples).
4. `atlas_tiny.py` (existing) and `atlas_small.py` modules:
   - Tiny: executes all supported algorithms per quantity to aid profiling, writes all resulting columns, and documents defaults.
   - Small: chooses the fastest algorithms from tiny benchmarks, writing only production columns.
   - Both rebuild datasets eagerly without resumable manifests.
5. Remove the legacy `atlas.py` references (already absent in the tree); any shared dataset I/O helpers should live in purpose-built modules (e.g., `io.py`) coexisting with the new helper stack.
6. Updated tests under `tests/viterbo/datasets2/` covering schema validation, converter round-trips, quantity wrappers, and smoke builds for both atlas variants.
7. Benchmarks in `tests/performance/viterbo/` (or updated existing ones) demonstrating build/runtime characteristics for tiny vs. small, with summary commentary in the brief or docs.
8. Documentation refresh: README snippets and a focused doc (e.g., `docs/datasets/atlas.md`) explaining the new workflow, module responsibilities, and how to extend columns.

## Execution Plan

1. Audit current `viterbo.datasets2` modules to catalogue existing schemas, converters, quantity functions, and plan abstractions slated for removal or relocation.
2. Implement (or extend) `schema.py` and `converters.py`, porting logic from `atlas_tiny.py` and any historical helpers, adding type hints, tests, and docstrings where necessary.
3. Build `quantities.py`, wrapping `viterbo.math` algorithms with explicit algorithm switches, column requirement checks, and shared error handling.
4. Extend `atlas_tiny.py` and implement `atlas_small.py` using straightforward loops over generators and quantities; ensure tiny registers multiple algorithms while small references the chosen production ones. Add an `io.py` (or similar) module if shared dataset I/O helpers are needed.
5. Rewrite smoke/unit tests to target the new helpers; add fixtures as needed and delete/adjust tests tied to deprecated abstractions.
6. Refresh benchmarks to exercise both atlas builders, capturing runtime deltas; update baseline artefacts or document rationale for changes.
7. Update documentation (README + dedicated datasets doc) with the new architecture, usage examples, and benchmark highlights; capture final notes in this brief.

## Open Questions & Risks

- No external consumers require compatibility shims for historical `build.py`/`atlas.py` modules; the new helpers can ship under `viterbo.datasets2` without re-export shims.
- Validate that benchmark environments are stable enough to compare tiny vs. small builds; if variance is high, add guidance on interpreting results.

