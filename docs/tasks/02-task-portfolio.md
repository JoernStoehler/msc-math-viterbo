# Task and Experiment Portfolio

This portfolio tracks the high-leverage work derived from `docs/algorithm-implementation-plan.md`
and the broader MSc thesis roadmap. Each item links to a dedicated brief inside `docs/tasks/`,
allowing Codex agents to pick up execution with minimal context switching while keeping
prioritisation visible at a glance.

## 1. Portfolio snapshot

| ID  | Brief                                                                                                    | Status      | Expected utility | Cost (agent / compute / expert) | Priority | Notes                                                                                            |
| --- | -------------------------------------------------------------------------------------------------------- | ----------- | ---------------- | ------------------------------- | -------- | ------------------------------------------------------------------------------------------------ |
| T1  | [Geometry quantity module restructure & JAX baselines](completed/2025-10-04-geometry-module-refactor.md) | Completed   | **+3.15**        | Medium / Low / Low              | —        | Root of SWE work; enables all downstream experiments.                                            |
| T2  | [Testing, benchmarking, and regression program](draft/2025-10-06-testing-benchmark-regression-program.md)        | In Progress | **+2.30**        | Medium / Low / Low              | 0.5      | Benchmarks and profiling hooks present; documentation and tiering remain.                        |
| T3  | [Symplectic invariants regression suite](draft/2025-10-06-testing-benchmark-regression-program.md)  | In Progress | **+2.50**        | Medium / Low / Medium           | 0.75     | Several invariants covered; baseline artefacts and docs pending.                                 |
| T4  | [JAX Pyright stub integration](scheduled/2025-10-06-jax-pyright-stubs.md)                                | In Progress | **+2.10**        | Medium / Low / Low              | 0.60     | Local stubs integrated; resolve JAX/NumPy type bridging to go green.                             |
| T5  | [Single JAX‑first LP solver](scheduled/2025-10-05-single-jax-first-lp-solver.md)                         | Scheduled   | **+2.00**        | Low / Low / Low                 | 0.55     | Simplifies onboarding; aligns with JAX‑first policy; replaces SciPy LP wrapper and abstractions. |
| E1  | [Facet-normal validation & dataset build](draft/2025-10-04-facet-dataset.md)                             | Draft       | **+2.70**        | Medium / Low / Low              | 1        | First numerical experiment; seeds data for the rest.                                             |
| E2  | [Reeb orbit cross-check](draft/2025-10-04-reeb-cross-check.md)                                           | Draft       | **+2.10**        | Medium / Low / Low              | 2        | Tests numerical agreement across methods.                                                        |
| E3  | [MILP relaxation bounds](draft/2025-10-04-milp-relaxations.md)                                           | Draft       | **+1.40**        | Medium / Medium / Medium        | 3        | Evaluates feasibility of open-source MILP tooling.                                               |
| E4  | [Capacity–volume correlation study](draft/2025-10-04-capacity-volume-study.md)                           | Draft       | **+1.80**        | Medium / Medium / Medium        | 4        | Mines dataset for trends and outliers.                                                           |
| E5  | [Support-function relaxation stress test](draft/2025-10-04-support-function-stress.md)                   | Draft       | **+0.90**        | Medium / Medium / Low           | 5        | Probes robustness of relaxation techniques.                                                      |

Expected utilities remain the sum of probability-weighted utilities captured in the owning briefs.
Priorities reflect the qualitative ordering: complete T1–T3 before launching the dataset or analysis
experiments.

## 2. Dependency structure

```mermaid
graph TD
  T1["T1\nGeometry module restructure"] --> T2["T2\nTesting & benchmarks"]
  T1 --> T3["T3\nInvariant regression suite"]
  T1 --> T4["T4\nJAX Pyright stubs"]
  T1 --> T5["T5\nJAX-first LP solver"]
  T2 --> T3
  T4 --> T3
  T3 --> E1["E1\nFacet dataset"]
  E1 --> E2["E2\nReeb cross-check"]
  E1 --> E3["E3\nMILP bounds"]
  E1 --> E4["E4\nCapacity-volume study"]
  E1 --> E5["E5\nSupport-function stress"]
  E3 --> E5
```

Update this graph whenever briefs change status or new items enter the queue.

## 3. Item summaries

### T1 — Geometry quantity module restructure & JAX baselines

- **Brief**:
  [`docs/tasks/completed/2025-10-04-geometry-module-refactor.md`](completed/2025-10-04-geometry-module-refactor.md)
- **Why it matters**: establishes the quantity-first package layout, harmonises JAX‑first
  reference/fast implementations with centralized wrappers, and seeds example datasets so later
  experiments can rely on consistent APIs.
- **Status**: Completed — downstream work (T2, T3) can now assume the new layout.

### T2 — Testing, benchmarking, and regression program

- **Brief**:
  [`docs/tasks/draft/2025-10-06-testing-benchmark-regression-program.md`](draft/2025-10-06-testing-benchmark-regression-program.md)
- **Why it matters**: merges the harness, marker taxonomy, and invariant coverage into one playbook so downstream experiments inherit trusted performance and correctness checks.
- **Next checkpoint**: fix the affine-map determinism test, confirm smoke runtime <3 min, then validate deep/longhaul tiers.

### T3 — Symplectic invariants regression suite

- **Brief**:
  [`docs/tasks/draft/2025-10-06-testing-benchmark-regression-program.md`](draft/2025-10-06-testing-benchmark-regression-program.md)
- **Why it matters**: shares the same consolidated program as T2, focusing on invariant baselines and regression coverage for downstream datasets.
- **Next checkpoint**: after smoke timing stabilises, extend baseline coverage and document escalation steps.

### T4 — JAX Pyright stub integration

- **Brief**:
  [`docs/tasks/scheduled/2025-10-06-jax-pyright-stubs.md`](scheduled/2025-10-06-jax-pyright-stubs.md)
- **Why it matters**: operationalises the stub strategy from RFC 002 so JAX-backed modules stay
  type-safe under Pyright strict.
- **Next checkpoint**: integrate the stub tree and supporting docs before resuming geometry
  enhancements.

### T5 — Single JAX‑first LP solver

- **Brief**:
  [`docs/tasks/scheduled/2025-10-05-single-jax-first-lp-solver.md`](scheduled/2025-10-05-single-jax-first-lp-solver.md)
- **Why it matters**: removes LP indirection and SciPy wrapper, standardises on a JAX‑native solver
  (JAXopt OSQP), and improves onboarding while keeping the library JAX‑first.
- **Next checkpoint**: implement `linprog_jax`, replace tests, and validate `just ci` locally; land
  as a single focused PR after current queue merges.

### E1 — Facet-normal validation & dataset build

- **Brief**: [`docs/tasks/draft/2025-10-04-facet-dataset.md`](draft/2025-10-04-facet-dataset.md)
- **Why it matters**: produces the canonical dataset feeding all other experiments, exercises
  algorithm variants, and surfaces anomalies early.
- **Next checkpoint**: choose storage format and confirm reproducibility before scaling to larger
  batches.

### E2 — Reeb orbit cross-check

- **Brief**:
  [`docs/tasks/draft/2025-10-04-reeb-cross-check.md`](draft/2025-10-04-reeb-cross-check.md)
- **Why it matters**: compares Reeb orbit estimators across algorithm families to highlight
  disagreements requiring theoretical attention.
- **Next checkpoint**: inventory available methods and ensure dataset interfaces cover their
  prerequisites.

### E3 — MILP relaxation bounds

- **Brief**:
  [`docs/tasks/draft/2025-10-04-milp-relaxations.md`](draft/2025-10-04-milp-relaxations.md)
- **Why it matters**: quantifies the value of open-source MILP tooling without relying on commercial
  solvers; informs whether tighter relaxations are worth pursuing.
- **Next checkpoint**: run pilot instances to confirm solver feasibility and log runtime before
  scaling.

### E4 — Capacity–volume correlation study

- **Brief**:
  [`docs/tasks/draft/2025-10-04-capacity-volume-study.md`](draft/2025-10-04-capacity-volume-study.md)
- **Why it matters**: searches for empirical structure in the dataset that could generate new
  conjectures or spotlight counterexamples.
- **Next checkpoint**: agree on statistical tooling and ensure notebook reproducibility fits within
  benchmark budgets.

### E5 — Support-function relaxation stress test

- **Brief**:
  [`docs/tasks/draft/2025-10-04-support-function-stress.md`](draft/2025-10-04-support-function-stress.md)
- **Why it matters**: tests robustness of relaxation strategies, feeding fixes or de-risking future
  experiments.
- **Next checkpoint**: gather scenarios from E1–E3 anomalies before executing the stress suite.

## 4. Long-haul benchmarking cadence

Long-haul benchmarks (>1 hour) should run **monthly** or before major milestones such as landing
T1/T2 or publishing new datasets. Capture outcomes by:

1. Saving raw measurements with `pytest --benchmark-autosave` (stored under `.benchmarks/`).
1. Adding a brief narrative (command, hardware, notable deltas) to the relevant task brief or weekly
   progress report immediately after the run so future work does not rely on memory.

Escalate to the maintainer if a long-haul run shows >10% regression or if hardware limits block
completion. For HPC/ML experiments include dataset scale, hardware type, precision, and random seeds
so future agents can reproduce the results or compare against CI artefacts.
