# Roadmap

_Last reviewed: 2025-10-02._

This roadmap balances literature work with incremental, testable Python code and reflects the current repository state.

## Current snapshot
- Symplectic capacity definitions, inequalities, and algorithm sketches are consolidated in the research notes on capacities and related docs, giving Phase 1 a solid reference base.【F:docs/13-symplectic-quantities.md†L1-L116】【F:docs/convex-polytope-cehz-capacities.md†L1-L73】
- The Python package now includes deterministic polytope constructors and transforms, a reproducible search enumerator, fast/reference volume backends, and a systolic-ratio wrapper on top of the EHZ implementations, providing the tooling for large-scale experiments.【F:src/viterbo/geometry/polytopes.py†L1-L200】【F:src/viterbo/optimization/search.py†L1-L100】【F:src/viterbo/geometry/volume.py†L1-L69】【F:src/viterbo/symplectic/systolic.py†L1-L80】
- Thesis scaffolding and the weekly mail workflow are in place, so planning tasks can tie directly into drafting and reporting cadence.【F:thesis/README.md†L1-L12】【F:docs/06-weekly-mail-workflow.md†L1-L27】

## Phase 1 — Foundations
Status: ✅ Core references drafted.
- Maintain the capacity definitions and reading list as new papers surface.
- Promote any ad-hoc notes from `tmp/` into the research docs to keep the foundation synced with the code base.

## Phase 2 — Techniques & Results
Status: 🟡 Iterating on coverage of results and algorithms.
- Expand the partial results section to highlight gaps the experiments could address.
- Capture benchmarking ideas for the algorithms already surveyed so they can graduate into reproducible tests later.
- Continue tagging open questions or proof sketches with TODO markers that link back to thesis chapters.

## Phase 3 — Python Implementation & Experiments
Status: 🟡 Core utilities landed; experiments need structuring.
- Harden performance-sensitive kernels with additional benchmarks and profiling hooks before scaling search runs.
- Integrate additional symplectic invariants (e.g., cylindrical capacity bounds) reusing the existing polytope abstractions.
- Add higher-level experiment scripts or notebooks that combine `search`, `volume`, and `systolic` helpers to generate reproducible datasets.

### Active milestone — Thesis-aligned experiment plan
- Sync the LaTeX outline with concrete computational objectives (link code modules to chapters/sections).【F:thesis/README.md†L1-L12】
- Define the first batch of search/volume experiments and record acceptance criteria directly in the roadmap so they feed the weekly progress workflow.【F:src/viterbo/optimization/search.py†L22-L91】【F:docs/06-weekly-mail-workflow.md†L1-L27】
- Ensure each experiment has an auditable path from raw polytopes to reported systolic ratios, including tests that guard against regressions.【F:src/viterbo/symplectic/systolic.py†L20-L80】【F:tests/symplectic/test_systolic.py†L1-L42】

## Phase 4 — Synthesis
Status: 🔜 Thesis drafting scaffolded, content to follow.
- Use the experiment outputs to populate thesis chapters and figures incrementally.
- Consolidate limitations, benchmarking results, and open questions alongside the narrative.

Tracking: Use GitHub Issues with the RFC template for larger design notes; smaller tasks as issues with clear acceptance criteria. Update the weekly mail drafts with roadmap deltas each Friday to maintain alignment.【F:docs/06-weekly-mail-workflow.md†L11-L27】
