# Task Brief — MILP Solver Selection (E3a)

- **Status**: Decision recorded (pending maintainer sign-off)
- **Last updated**: 2025-10-05
- **Owner / DRI**: Unassigned
- **Chosen solver**: HiGHS via `highspy`
- **Related docs**: `docs/tasks/02-task-portfolio.md`,
  `docs/tasks/draft/2025-10-04-milp-relaxations.md`, `docs/algorithm-implementation-plan.md`

## 1. Context and intent

The portfolio schedules an MILP relaxation experiment (E3) but the solver stack remains undecided.
We need a focused investigation to choose a single open-source MILP solver that balances
reliability, packaging, integration with our JAX-first workflow, and performance. The outcome will
unblock the MILP relaxation task and prevent churn in dependency management.

## 2. Objectives and non-goals

### In scope

- Survey candidate solvers (e.g., HiGHS via `highspy`, OR-Tools CP-SAT, CBC via PuLP/OR-Tools)
  across evaluation criteria: licensing, Python API ergonomics, installation complexity under `uv`,
  runtime stability on representative instances, and interoperability with JAX data pipelines.
- Prototype minimal MILP formulations (small facets dataset instances) to validate solver APIs and
  check deterministic behavior under CI constraints.
- Recommend a single solver for the repository, with justification, integration notes, and follow-up
  work (e.g., wrapping under `viterbo/_wrapped/`).

### Out of scope

- Implementing the full MILP relaxation task (E3) or benchmarking large instance suites.
- Supporting multiple solver backends or pluggable abstractions.
- Purchasing or evaluating commercial solvers.

## 3. Deliverables and exit criteria

- Decision note (Markdown under `docs/tasks/draft/`) summarizing evaluation, recommended solver, and
  integration guidance.
- Prototype script or notebook demonstrating solver usage on one or two polytopes from the facet
  dataset, checked into `tmp/` or `docs/tasks/draft/` alongside the note.
- Updated references in `docs/algorithm-implementation-plan.md` and task E3 brief to reflect the
  chosen solver.
- Checklist for dependency onboarding (pyproject entry, lockfile, CI considerations) ready for
  implementation task.

## 4. Dependencies and prerequisites

- Access to small representative MILP instances (can reuse E3 formulations at reduced scale).
- Coordination with maintainer for dependency policy confirmation (e.g., accepting `highspy` or
  `ortools`).
- Existing benchmarking harness (T2) for runtime sanity checks.

## 5. Execution plan and checkpoints

1. Inventory candidate solvers and collate metadata (license, maintenance status, packaging).
2. Implement minimal formulation testbed (shared across solvers) using existing dataset slices.
3. Run comparative trials (correctness, runtime footprint, install friction) on CPU-only
   environment.
4. Draft recommendation note with risks, migration steps, and integration checklist.
5. Review with maintainer; update portfolio/task briefs and close the investigation.

## 6. Evaluation matrix

Scoring uses a 1–5 scale (5 = best alignment). Weights sum to 1.0 and reflect the risks called out
in the brief and portfolio. Totals are the weighted sum of each solver’s scores.

| Candidate                | Packaging 0.20 | Runtime 0.25 | API 0.15 | JAX interop 0.15 | Typing 0.05 | Licensing 0.10 | Determinism 0.10 | Weighted total | Notes                                                                         |
| ------------------------ | -------------- | ------------ | -------- | ---------------- | ----------- | -------------- | ---------------- | -------------- | ----------------------------------------------------------------------------- |
| HiGHS (`highspy`)        | 5              | 5            | 4        | 4                | 3           | 5              | 4                | **4.50**       | MIT licence, manylinux wheels, mature simplex/MIP routines.                   |
| OR-Tools CP-SAT          | 4              | 4            | 3        | 3                | 2           | 4              | 3                | **3.50**       | Strong on integer-heavy models; weaker support for continuous relaxations.    |
| CBC via PuLP/`mip`       | 3              | 2            | 4        | 3                | 2           | 4              | 4                | **3.05**       | Friendly Python wrappers; performance tails off on dense MILPs.               |
| GLPK (`swiglpk`/`cvxpy`) | 2              | 2            | 2        | 3                | 1           | 1              | 4                | **2.20**       | GPL licensing complicates redistribution; slower on mixed-integer workloads.  |
| SCIP (ZIB licence)       | 2              | 5            | 3        | 3                | 1           | 1              | 4                | **3.10**       | Excellent solver tech but non-OSI licence imposes non-commercial restriction. |
| SYMPHONY                 | 2              | 2            | 2        | 2                | 1           | 3              | 3                | **2.15**       | Build-from-source flow; sparse ecosystem support and lower performance.       |
| lp_solve                 | 2              | 1            | 1        | 2                | 1           | 3              | 4                | **1.85**       | Legacy solver; limited API ergonomics and scalability.                        |

- HiGHS remains the clear front-runner: it dominates on performance and packaging while staying easy
  to integrate with lightweight wrappers.
- OR-Tools CP-SAT and CBC form the “usable” second tier—each has trade-offs (continuous support and
  packaging speed bumps, respectively) but can serve as fallback paths.
- GLPK, SYMPHONY, and lp_solve cover the legacy/open-source landscape yet introduce licensing or
  performance liabilities that conflict with the project’s goals.
- SCIP scores well technically but its non-OSI licence conflicts with the open-source-only scope;
  adopt only with maintainer-approved policy waiver.
- Commercial solvers (e.g., Gurobi, CPLEX, FICO Xpress) are excluded by policy but would otherwise
  outperform many open-source options; note their presence for completeness.

## 7. Decision summary

- Standardise on HiGHS via `highspy` for all MILP work in this repository.
- Treat OR-Tools CP-SAT and CBC as contingency solvers if HiGHS packaging or licensing changes.
- Record the decision in the E3 task brief and algorithm implementation plan before kicking off
  integration work.

## 8. Effort and resource estimates

- **Agent time**: Medium (≈ 0.5–1 agent-week)
- **Compute budget**: Low (CPU-bound micro instances)
- **Expert/PI involvement**: Low to Medium (policy confirmation, final solver approval)

## 9. Testing, benchmarks, and verification

- Use small MILP smoke tests to confirm API parity and determinism.
- Record runtime metrics (wall-clock, iterations) for comparison, but no exhaustive benchmarking
  required.
- Ensure prototypes run under `make pyright` and `make test` if committed.

## 10. Risks, mitigations, and escalation triggers

- **Risk**: Packaging issues (e.g., native dependencies) complicate CI. **Mitigation**: Prefer
  wheels available on PyPI; document build requirements early.
- **Risk**: Solver interfaces differ significantly, complicating reference integration.
  **Mitigation**: Scope prototypes to shared abstractions and note adapter requirements.
- **Escalation triggers**: Inability to install or run candidate solvers in the devcontainer/CI
  environment; unclear policy decisions on dependency footprint.

## 11. Follow-on work

- Update `docs/tasks/draft/2025-10-04-milp-relaxations.md` and
  `docs/algorithm-implementation-plan.md` to cite HiGHS as the committed solver.
- Work with the maintainer to approve dependency onboarding (add `highspy` to `pyproject.toml` and
  `uv.lock`, document platform notes).
- Implement solver integration under `_wrapped/` and extend MILP smoke/perf tests once the
  dependency lands.
- Document solver-specific tuning tips in `docs/` after adoption and surface any benchmark deltas in
  the weekly progress report or E3 findings.
