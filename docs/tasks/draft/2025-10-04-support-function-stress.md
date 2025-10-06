# Task Brief — Support-function relaxation stress test (E5)

- **Status**: Draft
- **Last updated**: 2025-10-04
- **Owner / DRI**: Unassigned
- **Related docs**: `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`

## 1. Context and intent

We plan to stress-test support-function-based relaxations using the curated dataset and insights
from MILP experiments. The goal is to understand when these relaxations remain trustworthy, identify
edge cases, and feed improvements back into algorithm design.

## 2. Objectives and non-goals

### In scope

- Select representative polytopes (including edge cases highlighted by earlier experiments).
- Implement evaluation harness comparing support-function relaxations with baseline capacities and
  MILP bounds.
- Investigate stability under perturbations (e.g., slight facet adjustments) and record failure
  modes.
- Recommend guardrails or heuristics to detect when relaxations are unreliable.

### Out of scope

- Designing entirely new relaxation families; focus on stress testing existing approaches.
- Heavy optimisation for runtime; leverage harness from T2 and reuse instrumentation.
- Publishing final conclusions before verifying anomalies with the maintainer.

## 3. Deliverables and exit criteria

- Scripts or notebooks executing the stress tests with reproducible configuration.
- Report summarising observed weaknesses, thresholds for reliability, and suggestions for
  algorithmic tweaks.
- Issues or follow-up tasks capturing required fixes or deeper investigations.

## 4. Dependencies and prerequisites

- Completion of E1 dataset; insights from E3 MILP relaxations highly beneficial.
- Benchmark harness and profiling tooling (T2).
- Availability of baseline support-function implementations in the restructured geometry modules.

## 5. Execution plan and checkpoints

1. **Scenario selection**: choose polytopes and perturbations based on previous anomalies.
1. **Harness setup**: reuse T2 instrumentation to compare relaxations vs. baselines.
1. **Stress execution**: run tests, capture metrics (accuracy, runtime, failure cases).
1. **Analysis**: cluster failure modes, inspect data, and hypothesise root causes.
1. **Reporting**: document results and propose follow-on tasks or mitigations.

## 6. Effort and resource estimates

- **Agent time**: Medium (≈ 0.5–1 agent-week)
- **Compute budget**: Medium (batch evaluations)
- **Expert/PI involvement**: Medium (interpretation of failure modes, prioritisation of fixes)

## 7. Testing, benchmarks, and verification

- Ensure CI smoke benchmarks cover at least one support-function scenario.
- Local deep benchmarks capture performance trends; long-haul runs optional when exploring severe
  perturbations.
- Add targeted unit tests reproducing discovered bugs or instabilities.

## 8. Risks, mitigations, and escalation triggers

- **Risk**: Stress tests reveal fundamental algorithm flaws requiring substantial redesign.
  **Mitigation**: Document clearly and triage follow-up tasks before attempting quick fixes.
- **Risk**: Perturbation space too large to explore exhaustively. **Mitigation**: Prioritise cases
  based on prior anomalies and theoretical interest.
- **Escalation triggers**: Discovery of systemic instability, need for new mathematical tooling, or
  compute budgets exceeding medium expectations.

## 9. Follow-on work

- Feed insights back into geometry module improvements and potential new relaxation tasks.
- Inform thesis chapters on limitations and future work.
