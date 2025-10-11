---
status: draft
created: 2025-10-11
workflow: task
summary: Outline placeholder notebooks and pipeline plan for the upcoming Monday meetup.
---

# Monday Meetup Visualisation Notebooks (Placeholder Plan)

## Context

- The PI requested placeholder notebooks to keep weekend efforts focused on the Monday meeting with the supervising professor.
- We need visual artefacts covering: (1) minimal action cycle projections for the Viterbo counterexample, (2) atlas-scale dimensionality reduction, (3) large polytope tables (enumerated, random, Mahler pairs), and (4) a shared dataset pipeline powering those views.
- Current helpers in `src/viterbo/` do not yet provide the required loaders, streaming, or plotting APIs; notebook code uses aspirational names only.

## Objectives

- Capture realistic execution sketches inside `.py` notebooks under `notebooks/proposed/`, emphasising data access patterns that scale beyond toy inputs.
- Document a dataset pipeline that coordinates chunked builds, caching, and manifest outputs required by the visual notebooks.
- Provide actionable follow-up checklists so future agents can turn placeholders into production-quality tooling.

## Execution

1. **Minimal action cycle notebook** — describe how to load the discrete trajectory, create synchronised 2D projections, and highlight labelled waypoints across the \(p\) and \(q\) components with a shared colour ramp.
2. **Atlas embedding notebook** — plan a stratified sampling strategy (≈20k points), streamed feature extraction, ANN-backed neighbour graphs, and a backend-agnostic embedding interface (UMAP/PaCMAP/TriMap).
3. **Polytope tables notebook** — outline manifest-style DataFrames for rotated \(n\)-gons, large random catalogues (50k+ entries), and Mahler pairs, keeping thumbnails as file references rather than in-memory arrays.
4. **Dataset pipeline notebook** — specify chunked, resumable builders with dedicated cache directories and provenance hooks to regenerate the above datasets without recomputation.
5. Track open questions directly in each notebook’s TODO section to drive Monday’s discussion.

## Dependencies / Unlocks

- **Upstream**: requires dataset builders, feature extractors, and visualisation utilities yet to be implemented in the main library.
- **Downstream**: once implemented, the notebooks will support automated report generation, counterexample discovery dashboards, and future clustering experiments.

## Status Log

- *2025-10-11*: Drafted placeholder notebooks and recorded execution constraints (sampling, streaming, caching) for later implementation.
