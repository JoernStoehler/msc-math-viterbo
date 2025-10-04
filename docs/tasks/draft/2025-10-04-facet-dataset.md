# Task Brief — Facet-normal validation and dataset build (E1)

- **Status**: Draft
- **Last updated**: 2025-10-07
- **Owner / DRI**: Unassigned
- **Related docs**: `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`

## 1. Context and intent
We aim to construct a curated dataset of polytopes, their facet normals, and derived systolic/capacity quantities to probe Viterbo's conjecture numerically. With the restructured geometry modules in place, this experiment validates the new APIs, checks consistency across algorithm variants, and produces artefacts for downstream analyses (E2–E5).

## 2. Objectives and non-goals

### In scope
- Assemble a diverse set of polytopes (simplices, cubes, zonotopes, stretched variants) using the new geometry package helpers.
- Compute facet normals, volumes, and baseline capacities using reference and optimised implementations.
- Validate invariants (e.g., transformations, symmetries) and document any discrepancies across implementations.
- Store the dataset in a reproducible format (e.g., JSON or Parquet) with metadata and versioning for future experiments.

### Out of scope
- Large-scale random sampling beyond the curated core set.
- Formal proof attempts or symbolic manipulations; focus on numerical artefacts.
- Publishing the dataset externally before internal validation completes.

## 3. Deliverables and exit criteria
- Dataset files stored under `data/` or another agreed location with README describing schema and generation scripts.
- Validation report summarising checks (transforms, cross-implementation comparisons) and anomalies.
- Updated task brief follow-up notes or weekly progress report capturing observed patterns, potential conjectures, and flagged edge cases.

## 4. Dependencies and prerequisites
- Completion of Task 2025-10-04-geometry-module-refactor — satisfied via the
  [completed brief](../completed/2025-10-04-geometry-module-refactor.md).
- Execution of Task 2025-10-04-testing-benchmark-harness
  ([scheduled brief](../scheduled/2025-10-04-testing-benchmark-harness.md)) to lock
  in regression coverage and benchmark markers.
- Agreement on dataset storage location and format.
- Benchmark cadence guidance from methodology to monitor runtime.

## 5. Execution plan and checkpoints
1. **Seed selection**: choose base polytopes and transformations to include.
2. **Pipeline scripting**: write deterministic generation code using reference algorithms, then verify against optimised/JAX variants.
3. **Validation sweep**: run invariance tests (e.g., rotations, scaling) and check cross-implementation parity.
4. **Packaging**: version dataset files, create README, and snapshot generation parameters.
5. **Review**: share summary with maintainer; decide whether dataset is internal-only or ready for sharing.

## 6. Effort and resource estimates
- **Agent time**: Medium (≈ 1 agent-week)
- **Compute budget**: Medium (batch runs across algorithms, CPU-bound)
- **Expert/PI involvement**: Low to Medium (review dataset semantics, highlight anomalies)

## 7. Testing, benchmarks, and verification
- Use unit tests from T2 harness to guard algorithm correctness.
- Add dataset-specific checks (schema validation, invariance tests) to CI if runtime permits; otherwise document manual steps.
- Record runtime metrics and store under `.benchmarks/` to inform future scaling.

## 8. Risks, mitigations, and escalation triggers
- **Risk**: Dataset generation uncovers inconsistencies between algorithm variants. **Mitigation**: Pause and open investigation issue before proceeding to subsequent experiments.
- **Risk**: Storage format insufficient for downstream tasks. **Mitigation**: Prototype with both JSON and Parquet; choose based on ergonomics.
- **Escalation triggers**: Lack of reproducibility (non-deterministic results), missing invariants, or compute cost ballooning beyond medium.

## 9. Follow-on work
- E2 (Reeb orbit cross-check), E3 (MILP relaxations), E4 (capacity-volume correlations), E5 (support-function stress tests).
