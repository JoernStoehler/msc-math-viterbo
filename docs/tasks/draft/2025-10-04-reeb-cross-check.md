# Task Brief — Reeb orbit cross-check (E2)

- **Status**: Draft
- **Last updated**: 2025-10-04
- **Owner / DRI**: Codex & PI (joint)
- **Related docs**: `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`

## 1. Context and intent

With the facet dataset in place, we want to cross-check Reeb orbit computations and symplectic
capacities derived from multiple algorithm families. This experiment probes discrepancies between
numerical approximations and theoretical expectations, informing where further mathematical
investigation is needed.

## 2. Objectives and non-goals

### In scope

- Implement or adapt Reeb orbit estimation routines compatible with the refactored geometry modules.
- Run the estimators across the curated dataset, recording variance between algorithm families.
- Identify cases where approximations diverge significantly and document hypotheses (numerical
  instability vs. modelling gaps).
- Propose diagnostics or new invariants that clarify discrepancies.

### Out of scope

- Developing entirely new Reeb orbit theory or proofs.
- Exhaustive optimisation for performance; focus on correctness and explainability.
- Publishing external notes before internal validation is complete.

## 3. Deliverables and exit criteria

- Scripts/notebooks that compute Reeb orbit quantities for each dataset element using multiple
  algorithmic approaches.
- Comparison tables highlighting agreements/disagreements, with narrative analysis.
- Issue list or TODOs capturing cases that require further investigation or follow-up tasks.

## 4. Dependencies and prerequisites

- Completion of E1 dataset build with validated artefacts.
- Access to geometry module variants and benchmark harness for runtime monitoring.
- Clarified invariants or theoretical expectations from algorithm plan or PI notes.

## 5. Execution plan and checkpoints

1. **Method inventory**: list available Reeb orbit estimation algorithms and their assumptions.
1. **Implementation pass**: ensure each algorithm integrates with the new geometry APIs.
1. **Batch evaluation**: run the dataset, capturing results and runtime metrics.
1. **Discrepancy analysis**: isolate notable divergences, inspect data, and hypothesise causes.
1. **Reporting**: summarise findings in the task brief or weekly progress report and propose
   follow-ups.

## 6. Effort and resource estimates

- **Agent time**: Medium (≈ 1 agent-week)
- **Compute budget**: Medium (batch evaluations)
- **Expert/PI involvement**: Medium (interpret anomalies, align with theoretical context)

## 7. Testing, benchmarks, and verification

- Unit tests comparing algorithm variants on known analytic cases (execute via `just test` smoke
  tier — 10 s per-test, 60 s session cap — before deep analyses).
- Integration tests ensuring dataset ingestion works end-to-end.
- Benchmark medium tier to ensure runtime stays within expectations; long-haul runs optional when
  exploring edge cases.

## 8. Risks, mitigations, and escalation triggers

- **Risk**: Numerical noise masks meaningful differences. **Mitigation**: Use high-precision
  arithmetic where possible; average across runs.
- **Risk**: API churn from upstream tasks. **Mitigation**: Coordinate with T1/T2 maintainers before
  refactors.
- **Escalation triggers**: Missing theoretical guidance, persistent unexplained discrepancies, or
  compute cost overruns.

## 9. Follow-on work

- Inform E3/E5 about corner cases needing stronger relaxations.
- Potential theoretical discussions with PI if counterexamples emerge.
