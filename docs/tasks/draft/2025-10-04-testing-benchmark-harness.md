# Task Brief — Testing, benchmarking, and profiling harness

- **Status**: Draft
- **Last updated**: 2025-10-05
- **Owner / DRI**: Unassigned
- **Related docs**: `docs/tasks/01-task-evaluation-methodology.md`, `docs/tasks/02-task-portfolio.md`, `docs/algorithm-implementation-plan.md`

## 1. Context and intent
After the geometry module restructure (Task 2025-10-04-geometry-module-refactor), we need automated verification to prevent regressions. This task establishes unit comparisons across implementation variants, codifies benchmark tiers (smoke, CI, deep, long-haul), wires profiling entry points, and ensures these hooks integrate with the existing `Makefile` + `pytest.ini` tooling so agents can reason about performance before launching computational experiments.

## 2. Objectives and non-goals

### In scope
- Build unit tests that compare reference, optimised, and JAX implementations on shared fixtures and transformation checks.
- Create pytest benchmark collections tagged by tier (`smoke`, `ci`, `deep`, `longhaul`) so CI and local runs can select the right scope quickly, including `pytest.ini` marker definitions and docstrings describing entry points.
- Add profiling recipes (e.g., `python -m cProfile`, `pyinstrument`) for the most critical code paths with instructions stored under `docs/` or `tests/performance/` and ready-to-run `uv run` commands.
- Document the three-tier cadence (fast CI <5 min, medium local 5–20 min, long-haul >1 h manual) and how to record results in PRs, task briefs, or weekly progress reports.

### Out of scope
- Major algorithmic rewrites or premature optimisation beyond instrumentation work.
- Adding GPU-specific benchmarking harnesses.
- Publishing benchmark dashboards; store raw data locally under `.benchmarks/` for now.

## 3. Deliverables and exit criteria
- New or updated tests ensuring algorithm variants agree on known fixtures and invariants.
- Benchmark suite organised with pytest markers and README guidance.
- Profiling how-to notes and scripts for the highest-priority modules.
- Documentation (could live alongside the benchmark README) explaining cadence expectations, how to invoke tiers via `make`/`pytest`/`uv run`, and where to log long-haul runs (task brief updates or progress reports).

## 4. Dependencies and prerequisites
- Completion of geometry module restructure (Task 2025-10-04-geometry-module-refactor) or a stable branch exposing reference/optimised/JAX variants.
- Access to curated fixtures delivered by the restructure task.
- Agreement on benchmark runtime targets (CI <5 min, deep local <20 min, long-haul manual) per evaluation methodology.
- Availability (or planned addition) of `pytest-benchmark` and profiling dependencies in `pyproject.toml`; escalate if missing.

## 5. Execution plan and checkpoints
1. **Fixture audit**: confirm the restructured modules expose canonical sample polytopes/volumes.
2. **Unit test sweep**: write or adapt tests that exercise each implementation variant side-by-side and register them with the new pytest markers.
3. **Benchmark scaffolding**: create pytest benchmark modules, mark tiers, update `pytest.ini`, and ensure the smoke tier runs within CI budget.
4. **Makefile integration**: extend or document `make bench`/related targets so each tier is easy to invoke locally and in CI.
5. **Profiling recipes**: draft scripts/notebooks for deeper inspection, capturing instructions in docs and verifying they run via `uv run`.
6. **Documentation pass**: update README or docs with cadence guidance and instructions for reporting results.
7. **Validation**: run `make ci` plus the deep benchmark tier locally; spot-check long-haul instructions by running at least one sample invocation for 5–10 minutes.

## 6. Effort and resource estimates
- **Agent time**: Medium (≈ 1 agent-week)
- **Compute budget**: Medium (due to benchmark runs, though still CPU-bound)
- **Expert/PI involvement**: Low (review cadence documentation and confirm instrumentation coverage)

## 7. Testing, benchmarks, and verification
- CI: `make ci` plus `pytest -m smoke_benchmark` (or equivalent marker) inside GitHub Actions.
- Local: run medium-depth benchmarks (`pytest -m deep_benchmark`) before publishing PRs touching performance-sensitive code.
- Manual: schedule long-haul benchmarks monthly or before major releases; log results in the task brief or relevant progress report.

## 8. Risks, mitigations, and escalation triggers
- **Risk**: Benchmarks exceed CI time budget. **Mitigation**: Keep smoke tier extremely small; gate larger sets behind markers.
- **Risk**: Profiling scripts drift from actual hot paths. **Mitigation**: Review instrumentation quarterly or when algorithms change materially.
- **Escalation triggers**: Missing fixtures, inability to stabilise benchmark runtime, or toolchain gaps (e.g., pytest-benchmark misconfiguration).

## 9. Follow-on work
- Dataset experiments (E1–E5) that rely on trustworthy baseline metrics.
- Future automation to publish benchmark trends if recurring runs justify dashboards.
