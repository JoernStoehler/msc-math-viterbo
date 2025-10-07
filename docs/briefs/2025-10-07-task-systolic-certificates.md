---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Produce rigorous systolic ratio certificates and conjecture statements based on computational evidence.
---

# Rigorous certificates and conjecture work

## Motivation

Following the ChatGPT Pro plan, we want machine-checkable evidence for \(\text{sys}(X)>1\) and structured conjectures grounded in data.

## Scope

- Build a certificate pipeline for rational polytopes, capturing Reeb orbit action bounds, exact volumes, and exhaustive search cutoffs.
- Articulate conjectures on subclasses (e.g., \(K \times K^\circ\), unconditional polytopes), backed by computed datasets.
- Analyse neighbourhoods around counterexamples to study stability of violations.

## Deliverables

1. JSON/PDF certificate format documenting orbit bounds and resulting inequalities.
2. Repository notes summarising conjectures, empirical support, and cross-links to relevant scans.
3. Tooling to regenerate certificates when polytope data updates.

## Dependencies & Links

- Consumes candidates and data from [Landscape scans and reproductions](2025-10-07-task-systolic-landscape-scans.md).
- Uses infrastructure from [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md) and theoretical guarantees from [Combinatorial Reeb formalization](2025-10-07-task-combinatorial-reeb-formalization.md).
- Provides labelled exemplars for [Machine learning program](2025-10-07-task-systolic-ml-program.md) and targets for [Advanced optimization directions](2025-10-07-task-systolic-advanced-directions.md).

## Status Tracking

Update the status when the certificate format or first certified counterexample lands; record conjecture revisions inline.

