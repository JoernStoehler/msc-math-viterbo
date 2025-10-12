---
status: retired
created: 2025-10-12
workflow: task
summary: Sketch the atlas-small dataset build and benchmarking run across generators and quantity algorithms.
---

# Subtask: Build atlas-small dataset and benchmark pipeline

## Context

- `viterbo.atlas` already exposes HF Datasets helpers and schema definitions, but no concrete atlas snapshot is shipped.
- Generator coverage currently spans random samplers in `viterbo.basic_generators` and structured enumerations; quantity algorithms live across `viterbo.capacity`, `viterbo.volume`, `viterbo.spectrum`, `viterbo.systolic`, and `viterbo.cycles`.
- The placeholder pipeline notebook (`notebooks/proposed/dataset_pipeline.py`) envisions chunked builders that log artefacts under `artefacts/`, but the updated scope treats the run as an in-memory pass with lightweight manifest-based resumability.

## Objectives

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

## Plan

1. **Scope generator coverage.** Catalogue every generator entry point (random samplers, structured enumerations, 4D products) and decide per-family parameter grids and sample counts for `atlas-tiny` (1 polytope) versus `atlas-small` (~10^3 polytopes overall). Capture this in a manifest draft so follow-on runs stay consistent.
2. **Design the build harness.** Extend the pipeline notebook or create a `scripts/build_atlas_small.py` wrapper that orchestrates generator invocation, quantity evaluation, and artefact storage. Ensure it reads/writes manifests, can filter to specific generators for resumability, and streams results directly to disk without chunk management overhead.
3. **Implement instrumentation.** Wrap each generator/quantity pair with timing and structured logging (JSON lines) that records success/failure, runtime, and traceback snippets for failures. Persist raw logs under `artefacts/benchmarks/atlas-small/`.
4. **Materialise `atlas-tiny`.** Run a smoke build that produces one polytope per generator, validate schema compatibility with `viterbo.atlas`, and document the <1 minute runtime target. Use this run to test resumability, logging, and the manifest workflow end-to-end.
5. **Produce `atlas-small`.** Scale the successful `atlas-tiny` configuration to the target sample counts, reusing the same harness and manifest. Store generated polytopes, computed quantities, and metadata in Hugging Face Dataset format; snapshot the final dataset manifest and version tag.
6. **Summarise benchmarks.** Aggregate timing logs into a Markdown/JSON report that highlights average runtimes, quantiles, and any failure counts per generator/quantity pair. Call out outliers and recommended follow-ups.
7. **Document consumption.** Update relevant docs/notebooks (e.g., dataset pipeline notebook, Monday notebook index) with loading instructions, dataset schema overview, and benchmarking findings. Include restart/resume guidance in the README accompanying the artefacts.

## Execution notes

- Prefer deterministic RNG seeds for random generators so repeated runs reproduce artefacts.
- Keep dataset shards under 100 MB to avoid git issues; use Git LFS if artefacts exceed the limit.
- End-to-end builds should stream results and clean up intermediates so the entire pipeline fits comfortably in memory without chunk scheduling.
- When algorithms fail or exceed runtime budgets, record the failure in the manifest and continue processing other items to keep the pipeline resilient.

## Decisions and constraints

- Generator coverage must span all available families: product n-gons with rational rotations, the random samplers, structured enumerations, and 4D products of random 2D polytopes.
- Quantities to compute for every polytope: raw geometry, generator metadata, dimension, vertex count, facet count, volume, EHZ capacity (covering each algorithm variant, favouring the fast-but-equivalent implementations), systolic ratio, and minimal action cycles. Slow or flaky algorithms should be profiled and documented rather than skipped.
- Runtime budget: target <10 minutes on the standard dev CPU for `atlas-small`; provide an `atlas-tiny` (<1 minute) dataset with one polytope per generator for smoke tests.
- Persist raw timing/instrumentation outputs alongside the processed benchmark summary so deeper analyses do not require reruns.
- Geometry artefacts can live in-repo (Git LFS is fine) without licensing concerns.

## Outcome

- Implemented `viterbo.atlas_build` with generator manifests, quantity instrumentation, and resumable execution for `atlas-tiny`
  and `atlas-small` presets.
- Added `scripts/build_atlas_small.py` to orchestrate dataset builds and benchmark logging into `artefacts/datasets/` and
  `artefacts/benchmarks/`.
- Created smoke tests that cover dataset generation, manifest integrity, timing logs, and resume semantics.
- Default plan spans five generator families with ~10³ samples for `atlas-small`; overrides allow lighter developer runs while
  retaining deterministic seeds.
- Dataset artefacts now include `manifest.json`, timing summaries, and README guidance for consumption and resumption.

## Open Questions

1. (Resolved) Adopted a purpose-built harness (`viterbo.atlas_build`) with JSONL logging; integration with `tests/performance`
   remains a potential follow-up if shared abstractions emerge.

## Notes

- Document any deviations from the stated generator/algorithm coverage so future runs can reconcile differences.
