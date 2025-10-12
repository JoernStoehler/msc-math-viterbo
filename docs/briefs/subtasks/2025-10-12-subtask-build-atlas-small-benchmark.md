---
status: draft
created: 2025-10-12
workflow: task
summary: Sketch the atlas-small dataset build and benchmarking run across generators and quantity algorithms.
---

# Subtask: Build atlas-small dataset and benchmark pipeline

## Context

- `viterbo.atlas` already exposes HF Datasets helpers and schema definitions, but no concrete atlas snapshot is shipped.
- Generator coverage currently spans random samplers in `viterbo.basic_generators` and structured enumerations; quantity algorithms live across `viterbo.capacity`, `viterbo.volume`, `viterbo.spectrum`, `viterbo.systolic`, and `viterbo.cycles`.
- The placeholder pipeline notebook (`notebooks/proposed/dataset_pipeline.py`) envisions chunked builders that log artefacts under `artefacts/`.

## Objectives (initial draft)

- Materialise a proof-of-concept `atlas-small` dataset (~10^3 polytopes) that exercises “nearly all” supported generators and invariants.
- Time each generator/quantity combination to surface bottlenecks and errors; emit an ad-hoc benchmark report alongside the dataset artefact.
- Keep the run resumable and idempotent so follow-up tasks can scale it.

## Deliverables (tentative)

- Dataset snapshots saved under `artefacts/datasets/atlas-small/` (layout TBD) plus manifest/metadata, with an accompanying `atlas-tiny` smoke dataset (<1 minute runtime) covering one polytope per generator.
- Timing report (Markdown or JSON) summarising runtimes and failures per algorithm.
- Update to documentation/notebooks pointing consumers at `atlas-small`.

## Dependencies

- Requires access to the generator implementations under `src/viterbo/basic_generators` and related modules.
- Requires quantity algorithms from `src/viterbo` (capacity, volume, spectrum, systolic, cycles) to be stable enough for batch execution.
- Feeds downstream consumers including the Monday notebooks and neural-encoding exploration subtasks once published.

## Acceptance criteria (to validate completion)

- `atlas-small` and `atlas-tiny` datasets exist under `artefacts/datasets/` with manifests that describe generator provenance, computed quantities, and schema versioning.
- Benchmark artefacts include both raw timing logs and a summarised report that identifies slow paths and any failures or fallbacks.
- Documentation or notebook references instruct contributors how to load the new datasets without additional context.
- The build process is idempotent and documents how to resume partial runs.

## Decisions and constraints

- Generator coverage must span all available families: product n-gons with rational rotations, the random samplers, structured enumerations, and 4D products of random 2D polytopes.
- Quantities to compute for every polytope: raw geometry, generator metadata, dimension, vertex count, facet count, volume, EHZ capacity (covering each algorithm variant, favouring the fast-but-equivalent implementations), systolic ratio, and minimal action cycles. Slow or flaky algorithms should be profiled and documented rather than skipped.
- Runtime budget: target <10 minutes on the standard dev CPU for `atlas-small`; provide an `atlas-tiny` (<1 minute) dataset with one polytope per generator for smoke tests.
- Persist raw timing/instrumentation outputs alongside the processed benchmark summary so deeper analyses do not require reruns.
- Geometry artefacts can live in-repo (Git LFS is fine) without licensing concerns.

## Open Questions

1. Select or design the timing harness: weigh integrating with existing benchmark tooling (e.g. `tests/performance`) versus a purpose-built notebook logger tailored for occasional atlas builds.

## Notes

- Document any deviations from the stated generator/algorithm coverage so future runs can reconcile differences.
