---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Cohesive milestone group translating ChatGPT Pro planning notes into actionable briefs on probing Viterbo's conjecture computationally.
---

# Systolic Ratio Exploration Overview

This bundle records the project outline suggested by the 2025-10-07 ChatGPT Pro session focused on building computational and mathematical tooling around systolic ratios of 4D polytopes. Each linked brief elaborates one strand of work; together they form an end-to-end exploration program from baseline validation through machine-driven discovery.

## Goal

Coordinate the milestones that will let us compute, certify, and explore systolic ratios for convex polytopes in \(\mathbb{R}^4\), including data-driven hypothesis generation and rigorous verification.

## Brief Map

- [Baseline invariants and regression tests](2025-10-07-task-systolic-baselines.md) – foundational checks.
- [Combinatorial Reeb model formalization](2025-10-07-task-combinatorial-reeb-formalization.md) – mathematical backbone.
- [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md) – engineering workhorse.
- [Landscape scans and reproductions](2025-10-07-task-systolic-landscape-scans.md) – initial computational experiments.
- [Rigorous certificates and conjecture work](2025-10-07-task-systolic-certificates.md) – mathematical deliverables.
- [Machine learning surrogate program](2025-10-07-task-systolic-ml-program.md) – data-driven acceleration.
- [Advanced optimization and analysis directions](2025-10-07-task-systolic-advanced-directions.md) – supplementary investigations.

## Dependencies & Order

1. Establish baselines before touching other strands.
2. Formalize the combinatorial Reeb model in tandem with the systolic pipeline; both depend on the baseline tests.
3. Landscape scans require the pipeline plus Reeb model formalization.
4. Rigorous certificates build atop scans and the pipeline, feeding back into baselines when new counterexamples emerge.
5. Machine learning consumes the atlas from landscape scans and certificate outputs for labels.
6. Advanced directions (shape derivatives, optimization, TDA, dynamical convexity, literature sweeps) rely on earlier milestones but can run in parallel once baselines, pipeline, and scans exist.

## Notes

- Track completion of this group via this overview; mark each linked brief's status when started or finished.
- Follow-up task: compare these proposals against existing implementations under `src/viterbo/` to identify overlaps or gaps.

