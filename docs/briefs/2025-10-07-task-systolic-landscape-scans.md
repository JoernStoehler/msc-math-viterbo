---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Run computational experiments to map systolic behaviour and reproduce literature results.
---

# Landscape scans and reproductions

## Motivation

The ChatGPT Pro milestones call for replicating Chaidez–Hutchings and Haim-Kislev–Ostrover experiments, then exploring broader polytope families to visualise where Viterbo’s conjecture fails.

## Scope

- Reproduce published datasets (Chaidez–Hutchings examples with \(\text{sys}=1\); the counterexample \(P\) with \(\text{sys}(P)>1\)).
- Generate landscape scans over structured families (e.g., \(K \times K^\circ\), unconditional polytopes, random normals) with histogram and action spectrum summaries.
- Track which combinatorial orbit types realise \(\mathrm{A}_{\min}\) across the scans.

## Deliverables

1. Notebooks or scripts producing replication tables with error bars.
2. Data artefacts (CSV/Parquet) capturing scans, ready for downstream ML.
3. Visual summaries illustrating distributions and qualitative phenomena.

## Dependencies & Links

- Requires [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md) and [Combinatorial Reeb formalization](2025-10-07-task-combinatorial-reeb-formalization.md).
- Uses fixtures from [Baseline invariants](2025-10-07-task-systolic-baselines.md).
- Feeds [Rigorous certificates and conjecture work](2025-10-07-task-systolic-certificates.md) with candidate polytopes.
- Supplies datasets for [Machine learning program](2025-10-07-task-systolic-ml-program.md) and diagnostics for [Advanced optimization directions](2025-10-07-task-systolic-advanced-directions.md).

## Status Tracking

Update status as replication and scanning milestones complete; note any blocking literature clarifications.

