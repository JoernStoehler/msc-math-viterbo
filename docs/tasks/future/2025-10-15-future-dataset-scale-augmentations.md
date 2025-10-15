---
title: "Future: dataset scale and augmentations for ML"
created: 2025-10-15
status: idea
owner: TBD
priority: high
labels: [future, datasets, ml]
---

## Summary

Target a “moderately large” core dataset of ~1e5 4D polytopes to enable training neural surrogates without overfitting. On top, generate augmented variants via:

- Group actions: apply `Sp(2n)` transformations and translations to exploit symplectic invariance while enriching coverage.
- Local smooth moves: adjust vertices by small gradient‑informed steps that preserve combinatorics so first‑order derivatives remain informative.
- Noise with redundancy: insert superfluous vertices and then jitter all vertices slightly so key quantities (capacity, etc.) change minimally even without explicit derivative/combinatorics checks.

The goal is a rich training corpus with controlled invariances and local neighborhoods that support learning both predictors (capacity/ratio) and structure (similarity, topology) at scale.

## Acceptance Hints

- Core set generation defined (random + structured families), with target size ~1e5.
- Augmentation pipeline sketched (group actions, local moves, noise) and integrated into data tooling.
- Basic audits: invariance checks under `Sp(2n)`+translation, stability of quantities under small local moves.

