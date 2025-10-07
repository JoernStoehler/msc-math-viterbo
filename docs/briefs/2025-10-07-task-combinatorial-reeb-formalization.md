---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Document and implement the combinatorial Reeb framework for 4D polytopes.
---

# Combinatorial Reeb model formalization

## Motivation

The ChatGPT Pro session stresses that Chaidez–Hutchings’ combinatorial Reeb model is the mathematical backbone aligning polytope dynamics with smooth Reeb flows. Formalising it ensures our computations have a solid theoretical anchor.

## Scope

- Translate the combinatorial Reeb orbit construction (facet itineraries, action, Conley–Zehnder index) into repository documentation and code interfaces.
- Prove and test invariance properties (scaling, symplectic linear maps) at the model level.
- Encode the Minkowski billiard correspondence for Lagrangian products \(K \times T\) and map it to EHZ capacity computations.

## Deliverables

1. Design doc describing the discrete Reeb model and smoothing correspondence.
2. Reference implementation with unit tests powered by [Baseline invariants](2025-10-07-task-systolic-baselines.md).
3. Examples covering \(K \times K^\circ\) and other benchmark domains.

## Dependencies & Links

- Consumes regression fixtures from [Baseline invariants](2025-10-07-task-systolic-baselines.md).
- Required by [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md) and [Landscape scans](2025-10-07-task-systolic-landscape-scans.md).
- Supports certificate work in [Rigorous certificates and conjecture work](2025-10-07-task-systolic-certificates.md).

## Status Tracking

Update the header `status` when the design doc or implementation lands.

