# Task Brief — Testing, Benchmarking, and Regression Program (T2/T2a/T3)

- **Status**: Draft
- **Last updated**: 2025-10-06
- **Owner / DRI**: Unassigned
- **Supersedes**: `docs/tasks/scheduled/2025-10-04-testing-benchmark-harness.md`,
  `docs/tasks/draft/2025-10-05-benchmark-marker-strategy.md`,
  `docs/tasks/scheduled/2025-10-05-symplectic-invariant-regression-suite.md`
- **Related docs**: `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`,
  `docs/tasks/completed/2025-10-04-geometry-module-refactor.md`

## 1. Context and intent

The geometry module restructure unlocked reference/fast implementation pairs and reusable fixtures,
scaffolding for richer regression guarantees. Three drafts (T2, T2a, T3) separately addressed the
harness, marker taxonomy, and symplectic invariant coverage. This merged brief defines a single
program that delivers end-to-end confidence: correctness/unit tests comparing implementations,
benchmarks with predictable cadences, profiling entry points, and invariant regressions that catch
mathematical drift before large experiment tasks (E-series) consume new datasets.

## 2. Golden path criteria (weights)

- **Reliability (0.25)**: Deterministic, reproducible results that surface regressions quickly in CI
  and local loops.
- **Developer ergonomics (0.15)**: Frictionless workflows for agents; one-command invocations via
  `just`/`uv run` with clear guidance.
- **Observability (0.15)**: Artefacts (baselines, benchmark histories, profiles) that make failures
  diagnosable without bespoke tooling.
- **Maintainability (0.15)**: Minimal bespoke glue; configurations live in `pyproject.toml`,
  `Justfile`, or tests with clear ownership.
- **Popularity (0.15)**: Adoption across the ecosystem so new agents arrive with existing intuition
  and community support.
- **Opinionatedness (0.10)**: Enforces a single golden path so agents do not fragment across multiple
  patterns.
- **Implementation effort (0.05)**: Workload to land and sustain the solution within the next
  sprint.

Weights sum to 1.0 and inform the decision matrices below. Ratings use a 1–5 scale (5 is best).

## 3. Objectives and non-goals

### In scope

- Consolidate unit tests that compare reference and JAX-first fast implementations using shared
  fixtures and jaxtyping signatures.
- Define and implement a pytest marker taxonomy aligned with runtime tiers and `just` recipes.
- Curate benchmark suites under `tests/performance/` with autosave, smoke/CI checks, and guidance on
  deep and long-haul cadences.
- Capture profiling recipes for hot paths with reusable `uv run` commands and `just` recipes.
- Codify symplectic invariant regression tests, baselines, and escalation playbooks for failures.
- Document the full workflow so new agents can follow a single golden path for the next six months.

### Out of scope

- GPU- or accelerator-specific profiling/benchmarking pipelines.
- Dashboard publication or automated trend visualisations beyond local artefacts.
- Adding brand-new algorithmic features; focus is instrumentation and validation of existing code.

## 4. Decision matrices

### 4.1 Marker taxonomy and tier gating

Assumptions: ratings reflect ability to meet CI (<5 min), local deep (5–20 min), and manual
long-haul (>1 h) budgets; higher implementation scores mean lower effort/risk. Popularity captures
agent familiarity; opinionatedness reflects how clearly the workflow enforces one path.

| Option                                                                                         | Reliability (0.25) | Dev ergonomics (0.15) | Observability (0.15) | Maintainability (0.15) | Popularity (0.15) | Opinionatedness (0.10) | Implementation effort (0.05) | Weighted score |
| ---------------------------------------------------------------------------------------------- | ------------------ | --------------------- | -------------------- | ---------------------- | ----------------- | ----------------------- | ---------------------------- | -------------- |
| A. Two-tier markers (`smoke`, `deep`) with ad hoc long-haul scripts                            | 4                  | 5                     | 3                    | 4                      | 3               | 3                     | 4                            | 3.75           |
| B. Three-tier markers (`smoke`, `deep`, `longhaul`) + existing `benchmark`/`line_profile` tags | 5                  | 4                     | 5                    | 4                      | 4               | 5                     | 3                            | 4.45           |
| C. Path-based selection only (no new markers)                                                  | 2                  | 3                     | 2                    | 3                      | 2               | 1                     | 5                            | 2.35           |

**Decision (locked 2025-10-06)**: Option B. It maximises observability, reliability, and familiarity while keeping
effort manageable; long-haul markers integrate cleanly with scheduled jobs without bespoke scripts
and reinforce a single policy-compliant path.

### 4.2 Invariant baseline artefact storage

Artefacts capture expected numeric ranges for invariant regressions and should be diff-friendly while
remaining familiar and prescriptive for new agents.

| Option                                                           | Reliability (0.25) | Dev ergonomics (0.15) | Observability (0.15) | Maintainability (0.15) | Popularity (0.15) | Opinionatedness (0.10) | Implementation effort (0.05) | Weighted score |
| ---------------------------------------------------------------- | ------------------ | --------------------- | -------------------- | ---------------------- | ----------------- | ----------------------- | ---------------------------- | -------------- |
| A. JSON snapshots per invariant family under `tests/_baselines/` | 5                  | 4                     | 5                    | 4                      | 4               | 4                     | 3                            | 4.35           |
| B. Single YAML catalogue shared across suites                    | 4                  | 3                     | 3                    | 3                      | 3               | 3                     | 4                            | 3.30           |
| C. Rely on pytest-benchmark history only                         | 2                  | 3                     | 4                    | 2                      | 2               | 2                     | 4                            | 2.55           |

**Decision (locked 2025-10-06)**: Option A. JSON keeps schema explicit, aligns with common tooling, and is
opinionated enough to steer agents toward consistent extensions while staying diff-friendly.

### 4.3 Profiling workflow integration

| Option                                                                                                  | Reliability (0.25) | Dev ergonomics (0.15) | Observability (0.15) | Maintainability (0.15) | Popularity (0.15) | Opinionatedness (0.10) | Implementation effort (0.05) | Weighted score |
| ------------------------------------------------------------------------------------------------------- | ------------------ | --------------------- | -------------------- | ---------------------- | ----------------- | ----------------------- | ---------------------------- | -------------- |
| A. `uv run` wrappers with `just profile`/`just profile-line` targets invoking `pyinstrument`/`cProfile` | 4                  | 5                     | 4                    | 4                      | 4               | 5                     | 4                            | 4.25           |
| B. Direct `pytest --profile` invocation per developer without Justfile integration                          | 3                  | 3                     | 2                    | 4                      | 3               | 2                     | 5                            | 3.00           |
| C. Notebook-based profiling playbooks                                                                   | 2                  | 2                     | 3                    | 2                      | 2               | 1                     | 2                            | 2.05           |

**Decision (locked 2025-10-06)**: Option A. It balances ergonomics, familiarity, and maintainability, giving a
clear entry point that matches CI/local environments and discourages ad hoc variants.

## 5. Recommended direction (summary)

- Locked decision (2025-10-06): adopt the three-tier marker taxonomy (`smoke`, `deep`, `longhaul`) layered atop existing
  `benchmark`, `line_profile`, and `slow` markers; surface them in `pyproject.toml` and Justfile
  recipes (`just test`, `just test-deep`, `just test-longhaul`).
- Locked decision (2025-10-06): store invariant baselines as structured JSON per family, version-controlled under
  `tests/_baselines/`, with helper fixtures to load/compare expected ranges.
- Locked decision (2025-10-06): provide profiling entry points via `uv run`-backed `just` recipes that collect reports into
  `.profiles/` and document post-processing steps.
- Maintain benchmarks in `tests/performance/` with autosave enabled; `smoke` tier runs in CI, `deep`
  tier runs before merging substantial geometry changes, and `longhaul` runs on a scheduled cadence
  with results logged in weekly reports or task briefs.
- Ensure unit/regression tests use shared fixtures and jaxtyping annotations so fast/reference
  implementations stay aligned.

## 6. Deliverables and exit criteria

- Updated tests under `tests/` and `tests/performance/` covering implementation parity and invariant
  regressions, tagged with the new marker taxonomy.
- JSON baseline artefacts plus loader utilities and guidance for extending them.
- Pytest configuration updates (`pyproject.toml`) enumerating marker descriptions and default
  deselection behaviour for CI.
- Justfile recipes and documentation describing how to run smoke/deep/longhaul tests, benchmarks,
  and profiling sessions via `uv run` commands.
- A README (or doc section) explaining cadence expectations, artefact locations, escalation steps,
  and reporting templates for benchmark/invariant drift.
- Evidence of validation: `just ci`, `just bench`, and one sampled `just test-deep`
  run recorded in the brief before handoff.

## 7. Execution plan and checkpoints

1. **Fixture and module audit**: inventory reference/fast pairs, shared fixtures, and existing
   invariant tests; identify coverage gaps.
2. **Marker taxonomy implementation**: define markers in `pyproject.toml`, update `just` recipes, and
   tag representative tests/benchmarks to confirm selection semantics.
3. **Correctness sweep**: implement or refactor unit tests comparing reference vs fast paths with
   jaxtyping annotations and deterministic seeds.
4. **Invariant baseline capture**: codify additional invariants, capture expected values into JSON
   artefacts, and wire fixtures that diff results with helpful failure messages.
5. **Benchmark restructuring**: segment existing `tests/performance/` suite into smoke/deep sets,
   validate runtimes, and document autosave conventions.
6. **Profiling recipes**: create `uv run` wrappers and `just` recipes for `pyinstrument` and
   line-profiler flows; add documentation.
7. **Documentation pass**: consolidate instructions, cadence, and escalation details within the task
   brief/README; circulate for maintainer feedback.
8. **Validation**: execute CI-equivalent runs plus at least one deep suite and archive artefact
   samples in the task brief or linked reports.

## 8. Dependencies and prerequisites

- Geometry restructure (completed) providing canonical fixtures and reference/fast APIs.
- Agreement with maintainer on CI runtime budgets and long-haul scheduling cadence.
- Access to existing pytest-benchmark setup and profiling dependencies declared in `pyproject.toml`
  (dev extra).
- Coordination with dataset tasks (E1–E5) to ensure invariant coverage meets their preconditions.

## 9. Testing, benchmarks, and verification

- **CI**: `just ci` runs smoke-tier unit tests; `just bench` executes the CI
  benchmark slice. Failing invariants or capacity comparisons trigger escalation per documentation.
- **Local deep loop**: Developers run `just test-deep` and `just bench-deep` before
  landing geometry or optimisation changes; profiling via `just profile` when performance drift is
  suspected.
- **Long-haul**: Scheduled (weekly/monthly) `just test-longhaul` and `just bench-longhaul`
  runs capture artefacts stored under `.benchmarks/` and `.profiles/`, with summaries archived in
  progress reports.

## 10. Risks, mitigations, and escalation triggers

- **Risk**: Marker tier runtimes exceed budgets. **Mitigation**: measure after tagging; adjust test
  assignments before enabling in CI.
- **Risk**: Baseline JSON drifts due to floating-point sensitivity. **Mitigation**: document
  tolerances, use float64, and flag deviations for maintainer review instead of auto-updating.
- **Risk**: Profiling scripts go stale when APIs change. **Mitigation**: keep commands minimal,
  review quarterly alongside benchmark updates.
- **Escalation triggers**: CI smoke tier >5 minutes, inability to reproduce deep tier locally,
  disagreement on invariant definitions, or need for environment/toolchain changes beyond the
  provisioned stack.

## 11. Follow-on work

- Automate publication of benchmark/invariant trends once artefact history stabilises.
- Extend harness to GPU or distributed contexts if future tasks require it.
- Evaluate integration with external observability tools (e.g., W&B) once current efforts land.

## 12. Current status (2025-10-06)

- Repaired `tests/viterbo/geometry/polytopes/test_transforms.py::test_random_affine_map_is_deterministic_per_seed`; same PRNG key now produces identical samples while split keys may diverge.
- Measured smoke-tier modules individually (allocator kills any run >20% wall-time). Recorded wall-time per module with `JAX_ENABLE_X64=1`:
  - `tests/viterbo/geometry/polytopes/test_transforms.py` — 8.6 s (slowest cases: random polytope facets 5.3 s, deterministic affine map 1.4 s).
  - `tests/viterbo/geometry/polytopes/test_combinatorics.py` — 5.6 s.
  - `tests/viterbo/geometry/polytopes/test_haim_kislev_action.py` — 1.6 s.
  - `tests/viterbo/geometry/halfspaces/test_halfspaces.py` — 3.5 s.
  - `tests/viterbo/geometry/volume/test_volume.py` — 41.5 s (dominant case: random polytope volumes 33.8 s).
  - `tests/viterbo/symplectic/capacity/facet_normals/test_reference.py` — 11.1 s.
  - `tests/viterbo/symplectic/capacity/facet_normals/test_algorithms.py` — terminated at 13.3 min (largest case: cross-polytope-4d at 573 s) before allocator cut the run.
  - `tests/viterbo/symplectic/capacity/facet_normals/test_fast.py` — terminated at ~59 min (multiple cross-polytope variants at 11–12 min apiece) before allocator cut the run.
  - `tests/viterbo/symplectic/test_core.py` — 1.2 s.
  - `tests/viterbo/symplectic/systolic/test_systolic.py` — 5.4 s.
  - `tests/viterbo/optimization/test_search.py` — 55.3 s (enumerate search space deterministic case: 54.7 s).
  - `tests/viterbo/optimization/test_solvers.py` — 0.8 s.
- Full smoke collection remains unreachable without a higher wall-time allocation; every fresh `uv run pytest` incurs JAX compilation, so the allocator ends the long facet-normal suites before they complete.
- Deep tier, benchmark targets, and documentation updates are still pending on successful smoke aggregation.
- Re-tiered the worst offenders: facet-normal agreement sweeps now keep only lightweight 2D
  polytopes (simplex, hypercube, cross-polytope) in smoke; all catalog-derived cases stay in `deep`
  for both reference and fast implementations. The random polytope volume comparison is `deep`,
  and the search-space determinism test caps itself at 12 candidates. Re-run timings once the
  remaining smoke cases finish under the enforced 10 s per-test / 60 s session timeout. Added
  incremental loops via `just test-incremental` and a staged pre-commit workflow
  (`just precommit-fast` for quick passes, `just precommit-slow` before handoff).

### BOTEC: utility vs cost per representative test suite

| Test module | Primary failure mode caught | Utility (↑ better) | Maintenance complexity | CI wall-time (s) | Deep/bench impact | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `geometry/polytopes/test_transforms.py` | Math-level value errors (affine transforms, determinism), shape sanity | 4 | 2 | 8.6 | Low | Validates random map contract and linear algebra glue; moderate runtime. |
| `geometry/polytopes/test_combinatorics.py` | Math-level value errors (facet/vertex enumeration), combinatorial regressions | 4 | 3 | 5.6 | Low | Exercises enumerators against cached fixtures; good signal per second. |
| `geometry/volume/test_volume.py` | Algorithmic correctness for Monte Carlo/reference volumes | 5 | 3 | 41.5 | Medium | Detects drift in sampling/normalisation; expensive, consider deep-tier. |
| `symplectic/capacity/facet_normals/test_reference.py` | Math-level value errors vs baselines | 5 | 3 | 11.1 | Medium | High-fidelity guardrail; cost acceptable if kept deep. |
| `symplectic/capacity/facet_normals/test_algorithms.py` | Algorithmic regressions in fast vs reference kernels | 5 | 4 | >800 | High | Critical but prohibitively slow; best fit for deep/longhaul. |
| `symplectic/capacity/facet_normals/test_fast.py` | Fast-path numerical agreement across variants | 5 | 4 | >3500 | High | Exhaustive coverage; belongs in longhaul with profiling support. |
| `symplectic/systolic/test_systolic.py` | Mathematical invariants (scale/translation) | 3 | 2 | 5.4 | Low | Moderate utility; cost acceptable for smoke. |
| `optimization/test_search.py` | Algorithmic correctness (search enumeration) | 4 | 3 | 55.3 | Medium | Ensures deterministic coverage; runtime suggests deep tier. |
| `optimization/test_solvers.py` | Python/type-level wiring to external solvers | 2 | 2 | 0.8 | Low | Mostly guards argument plumbing; cheap. |

Utility heuristic (0–5): 5 = catches high-severity mathematical regressions; 1 = mostly style/typing noise. Maintenance complexity (0–5): 5 = fragile fixtures or heavy baselines; 1 = trivial unit tests. Deep/bench impact qualitatively notes whether reclassifying affects extended tiers.

- Takeaway (BOTEC): moving the long facet-normal and Monte Carlo suites entirely out of smoke would cut CI wall-time by ~65–70 s per run and avoid allocator kills, at the cost of deferring high-utility math regressions to deep/longhaul tiers. No changes executed yet—capturing measurements first to guide reprioritisation discussions.

#### PI guidance (2025-10-06)

- Enforce per-test wall-time ceilings in the smoke tier (e.g., pytest `timeout` + Justfile-level guards) so runaway cases are caught early; complement with a hard cap on overall `just test` runtime.
- Profile hotspots to ensure pure math kernels stay ≤500 ms; focus on JAX tracing reuse by stabilising static arguments (loop counts, polytope descriptors, PRNG branching). Check whether unrolled loops rely on static argnums, whether `Polytope` objects participate in pytree registration, and whether per-instance metadata is forcing recompilation—reason through `jax.jit` traces mentally before running them.
- Capture both compilation and execution durations explicitly—fine-grained profiling should separate XLA compile time from kernel runtime to highlight caching wins and quantify cold-start drag.
- For manual workflows, keep a warm interpreter (e.g., Jupyter notebook or REPL) to avoid cold-start JAX cost when iterating on tests/profiling.
- Audit stochastic tests for unnecessary loop counts (e.g., 1,000 random polytopes) and replace them with explicit adversarial fixtures where known; the random sampling is there to dodge human blind spots, not to discharge conjectures.
- When randomness remains, report failure density in looped tests (first failing iteration counts) to quantify how quickly issues surface and justify the remaining stochastic budget.
- Continue reevaluating marker assignments with the above data; migrate high-utility but heavyweight suites towards deep/longhaul tiers once alternative fast guards exist.


## 13. Agent handoff prompt

Next agent: secure a wall-time allocation that can sustain the long facet-normal suites (or temporarily reclassify them) so a full smoke collection can finish in one pass. Once end-to-end smoke timings are captured, proceed with `just test-deep`, `just bench`, and `just bench-deep`, then propagate the consolidated measurements and any marker adjustments back into §12.
