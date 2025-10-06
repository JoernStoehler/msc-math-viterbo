# Task Brief — MILP relaxation bounds (E3)

- **Status**: Draft
- **Last updated**: 2025-10-04
- **Owner / DRI**: Unassigned
- **Related docs**: `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`

## 1. Context and intent

Building on the facet dataset, this experiment explores MILP relaxations that bound symplectic
capacities. The goal is to gauge feasibility with open-source solvers, quantify gaps between
relaxations and baseline capacities, and determine whether tighter formulations are worth pursuing.

## 2. Objectives and non-goals

### In scope

- Formulate MILP problems aligned with existing conjectures using the standardised HiGHS (`highspy`)
  solver.
- Run experiments on dataset instances to observe relaxation quality and solver runtime.
- Compare bounds against reference capacities to measure tightness and detect counterexamples.
- Document solver performance characteristics (iteration counts, failure cases) for future
  optimisation work.

### Out of scope

- Procuring commercial solvers; the team decided to stay open-source.
- Large-scale cluster runs; keep workloads manageable on available hardware.
- Rigorous proof attempts derived from MILP outcomes (can inspire but not replace formal work).

## 3. Deliverables and exit criteria

- Code or notebooks setting up MILP formulations, with instructions for reproduction.
- Results tables summarising bound quality per polytope, solver runtimes, and notable failures.
- Recommendations on whether further MILP investment is worthwhile (e.g., refine formulations vs.
  pivot to alternative methods).

## 4. Dependencies and prerequisites

- Completion of E1 dataset and availability of baseline capacities for comparison.
- Benchmark harness to monitor solver runtime within acceptable budgets.
- Familiarity with the HiGHS (`highspy`) solver API; contingency solvers (OR-Tools CP-SAT, CBC)
  remain optional fallbacks but are out of scope unless HiGHS becomes unavailable.
- Policy alignment: keep non-JAX solver interop out of library code. If any MILP code graduates
  beyond notebooks/scripts, isolate it under `viterbo/_wrapped/` per AGENTS policy. Propose
  environment additions via maintainer approval before introducing new solver dependencies.

## 5. Execution plan and checkpoints

1. **Formulation review**: adapt existing theorems/definitions into MILP-friendly constraints.
1. **Prototype implementation**: encode problems for a small subset of polytopes; validate
   feasibility.
1. **Solver commitment**: integrate the HiGHS backend and run convergence/runtime checks on
   representative instances to confirm parity with reference relaxations.
1. **Analysis**: compare bounds vs. baseline capacities, highlight gaps or counterexamples.
1. **Reporting**: compile findings into the task brief or weekly progress report and recommend next
   steps.

## Open questions

- Ensure the HiGHS via `highspy` integration remains healthy (pin version, document options) and
  capture any regressions or limitations that might require contingency solvers.

## 6. Effort and resource estimates

- **Agent time**: Medium (≈ 1–1.5 agent-weeks)
- **Compute budget**: Medium (solver runs can be CPU-intensive but manageable)
- **Expert/PI involvement**: Medium (interpretation of relaxation quality and theoretical
  implications)

## 7. Testing, benchmarks, and verification

- Unit tests for MILP encoding functions to ensure constraint matrices match expectations.
- Small smoke problems executed in CI to guarantee formulations remain feasible.
- Local deep runs for more complex instances; document runtime and solver diagnostics.

## 8. Risks, mitigations, and escalation triggers

- **Risk**: MILP formulations too loose to be informative. **Mitigation**: iterate quickly on
  constraints; abandon if bounds remain weak.
- **Risk**: Solver instability on certain instances. **Mitigation**: switch solvers, adjust
  tolerances, or precondition inputs.
- **Escalation triggers**: Need for solver-specific expertise, repeated infeasibility without clear
  cause, or runtime exceeding agreed medium budget.

## 9. Follow-on work

- If successful, feed tighter bounds into E5 stress tests or suggest formal proof avenues.
- If not, pivot to alternative relaxations (e.g., SDP) or reinforce dataset diagnostics.
