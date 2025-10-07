---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Build reference polytopes and invariance tests to anchor systolic computations.
---

# Baseline invariants and regression tests for systolic ratio tooling

## Motivation

Reproducing the simple families from the ChatGPT Pro plan (ellipsoids, products, symplectic transforms) gives us a trusted yardstick before investing in heavier computations.

## Scope

- Implement sanity checks for \(\text{sys}(X)\) invariance under scaling and linear symplectomorphisms.
- Produce dual H/V representations and confirm consistent volumes and capacities.
- Curate a reference library of centrally symmetric 4D polytopes, including Lagrangian products of 2D polygons, to ship with the repository.

## Deliverables

1. Automated test suite covering ellipsoids and product domains with known \(\text{sys}(X)\) values.
2. Reproducible scripts/data for the reference polytope set.
3. Documentation summarising expected outputs and tolerances.

## Dependencies & Links

- Enables [Combinatorial Reeb model formalization](2025-10-07-task-combinatorial-reeb-formalization.md) and [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md).
- Feeds regression fixtures for [Landscape scans and reproductions](2025-10-07-task-systolic-landscape-scans.md).

## Status Tracking

Status stays `proposed` until tests and data are merged; update this file when work starts or completes.

