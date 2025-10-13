---
status: draft
created: 2025-10-12
workflow: task
summary: Frame a research sweep over neural encodings and architectures for polytope data.
---

# Subtask: Explore neural encodings for polytopes

## Context

- Upcoming ML experiments will rely on atlas-style datasets containing polytope geometry (vertices, halfspaces) and derived invariants.
- Potential modelling strategies include graph neural networks, transformers over set representations, and feature-enriched baselines.
- Existing documentation mentions brainstorming but lacks a concrete evaluation framework.

## Objectives (initial draft)

- Catalogue candidate data representations (graphs, sequences, tensors) for polytopes and map them to neural architectures.
- Identify preprocessing/enrichment steps (e.g. symplectic invariants, canonical orderings) that improve learnability.
- Deliver a prioritised plan for prototyping, including metrics, datasets, and tooling requirements.

## Deliverables (tentative)

- Research brief or design doc summarising representation options, pros/cons, and recommended next experiments.
- Supporting notebook or script illustrating at least one encoding pipeline (if feasible within scope).
- List of open research questions and dependencies on dataset/tooling workstreams.

## Dependencies

- Consumes atlas datasets (starting with the existing `atlas_tiny` snapshot); larger presets such as `atlas_small` depend on the builder expansion work.
- Coordinates with the Monday notebooks for visual outputs and with dataset benchmarks for feature availability.
- Requires access to baseline classical algorithms to establish comparative metrics.

## Acceptance criteria (to validate completion)

- The research brief documents at least three encoding strategies, articulating their data requirements, architectural fit, and evaluation plans.
- Proposed experiments specify metrics, datasets, and tooling stacks (including any external libraries) with reproducibility guidance.
- At least one illustrative prototype (notebook or script) demonstrates data preparation for a candidate architecture or baseline method.
- Outstanding research questions and next actions are catalogued with owners or follow-up subtask suggestions.

## Decisions and constraints

- Exploration may include prototypes or stubs when they clarify feasibility, but polished implementations are optional.
- Downstream tasks to prioritise: capacity prediction, classification, and clustering derived from atlas invariants.
- PI will provide existing notes at task kickoff—remember to request them explicitly.
- Plan for an immediate-sprint roadmap that can run continuously; keep momentum without waiting for extra approvals.
- Include reproducible non-neural baselines (e.g. linear or classical methods) alongside neural proposals.
- Preferred tooling: JAX and SciPy, leveraging HF Datasets loaders; introduce other libraries only with follow-up justification.

## Open Questions

1. None outstanding pending receipt of the PI's prior notes.

## Notes

- Encourage contributors to store exploratory notebooks under `notebooks/research/` with clear naming and metadata cells.
