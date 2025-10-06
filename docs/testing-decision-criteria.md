# Test Selection Decision Criteria

This document explains how we decide which tests to create, how to prioritise them across the
available runtime tiers, and which supporting tooling (typing, benchmarks, profiling, coverage)
should accompany each proposal. The goal is to make test planning repeatable: given a new piece of
behaviour, contributors can consult these criteria to evaluate whether a test is worth the effort,
how it should be scoped, and where it belongs in the validation pipeline.

## 1. Quality dimensions we guard

We add or adjust tests to defend one or more of the following dimensions. Every new proposal should
state which dimension(s) it targets and why existing guards are insufficient.

- **Logical correctness**: ensures control flow and algebraic relations execute without runtime
  errors. Fast feedback sources include static analysis (pyright) and small deterministic unit tests.
- **Numerical fidelity**: verifies that algorithms converge to the expected value on trusted
  datasets. Usually requires larger fixtures or comparisons against reference implementations.
- **Performance envelopes**: confirms that we meet runtime, compilation, and memory budgets. Bench
  suites and profiling flows are the main tools here.
- **API stability & usability**: catches breaking changes by forcing representative usage patterns to
  remain easy to express. Regression tests double as executable documentation.
- **Observability & artefacts**: validates that invariant baselines, benchmark histories, and profiles
  stay in sync with the code and provide actionable failure messages.

## 2. Decision workflow for new or revised tests

Follow these steps when triaging a proposed test case. Iterate until each criterion is satisfied or
we discard the proposal.

1. **Clarify the failure mode**
   - What risk are we defending? (e.g., incorrect math, unstable optimisation, slow kernel,
     accidental API change.)
   - How severe is the impact if it ships? Severity guides how eager we are to invest runtime and
     maintenance.

2. **Check existing coverage**
   - Do current tests, property checks, type hints, or benchmarks already detect this risk?
   - Can we strengthen an existing guardrail instead of adding a new one?
   - If we rely on static analysis alone, is there a runtime behaviour that still escapes detection?

3. **Select the evidence type**
   - **Static typing / linting**: best for interface misuse and shape mismatches. Prefer improving
     type annotations before adding slow runtime checks for the same issue.
   - **Smoke-tier runtime tests**: cover deterministic, low-cost scenarios that catch regressions in
     <5 minutes wall-time. Use them for high-frequency bug classes with small fixtures.
   - **Deep-tier runtime tests**: run heavier deterministically reproducible cases (5–20 minutes) to
     compare reference vs fast paths, cover larger datasets, or explore corner cases that are too
     heavy for smoke.
   - **Longhaul tests**: reserved for exhaustive sweeps or stochastic integrations that need more than
     20 minutes. Require clear escalation criteria and artefact capture.
   - **Benchmarks**: measure runtime budgets on representative workloads. Require baseline storage and
     a plan for interpreting regressions.
   - **Profiling entry points**: only when we have a known hotspot that needs instrumentation to stay
     within performance envelopes.

4. **Decide tier placement**
   - Estimate runtime (cold and warm) with x64 enabled and decide which tier budget it fits.
   - Smoke tests should finish in under ~30 seconds each and avoid large compilation overheads.
   - Deep tests may tolerate slower cases but must remain reproducible and provide deterministic
     output.
   - Longhaul tests must deliver insights that justify manual scheduling (e.g., high-confidence
     numerical baselines or exhaustive combinatorics).

5. **Choose fixtures and parameters**
   - Prefer deterministic fixtures that target the failure mode directly. Use randomness only when it
     uncovers classes of bugs we cannot enumerate manually.
   - When randomness is necessary, fix seeds and record failure density (e.g., first failing iteration)
     to justify the sample count.
   - Scale input sizes to the smallest data that still triggers the behaviour. Document why each
     parameter (iteration counts, grid sizes, tolerances) is required.

6. **Design assertions and diagnostics**
   - Ensure failure messages explain what regressed (value bounds, invariants, tolerances).
   - For numerical comparisons, set tolerances deliberately and note the rationale (conditioning,
     baseline variability).
   - For performance tests, capture both compilation and execution times separately when practical.

7. **Evaluate maintenance cost**
   - Consider fixture complexity, baseline updates, and coupling to implementation details.
   - Avoid adding tests that demand frequent manual intervention unless the guarded risk is critical.
   - Prefer reusing shared fixtures and helper utilities to reduce duplication.

8. **Plan artefact integration**
   - Decide where outputs live (`.benchmarks/`, `.profiles/`, JSON baselines). Ensure Make targets and
     documentation point to them.
   - Set escalation triggers (e.g., ±10% runtime regression, invariant mismatch) and document the
     response plan.

9. **Document the outcome**
   - Record the decision (add/modify/reject) in the relevant task brief or PR description.
   - Update workflow documentation if the decision introduces a new pattern other agents must follow.

## 3. Comparing alternative proposals

When we have multiple ways to cover the same risk, score each option against these axes and pick the
combination with the highest overall value within our runtime budget.

- **Signal quality**: probability of detecting the targeted regression. Prefer precise fixtures with
  narrow failure scopes over broad but flaky stochastic sweeps.
- **Runtime cost**: cold-start vs warm execution, and how often the tier runs (per commit, pre-merge,
  scheduled).
- **Implementation effort**: complexity of fixtures, need for new infrastructure, and reviewer burden.
- **Maintenance burden**: expected churn on baselines, tolerance updates, or dataset refreshes.
- **Diagnostic clarity**: how easily developers can interpret a failure and act on it.
- **Complementarity**: coverage diversity. Two cheap tests that fail on the same condition might be
  redundant; a single deeper test that exercises different code paths could be better.

We prefer a solution set that maximises signal quality and complementarity while keeping runtime cost
and maintenance manageable. If a proposal scores poorly on runtime but strongly on signal quality,
consider demoting it to a slower tier or refactoring the fixture to a reduced yet still effective
variant.

## 4. Parameters and knobs worth tuning

For each test we evaluate:

- **Input size / iteration counts**: lower them until further reductions would stop triggering the
  targeted behaviour.
- **Precision / dtype**: default to float64; only lower precision when we explicitly test mixed-precision
  behaviour and document the rationale.
- **Seed management**: hold seeds locally inside tests, split keys when parallelism is required, and
  avoid global PRNG state.
- **Tolerance levels**: choose `rtol`/`atol` based on problem conditioning; explain any deviation from
  default strict tolerances.
- **Timeouts**: apply per-test timeouts for smoke-tier cases to prevent runaways, complementing tier
  wall-time caps.
- **Marker assignments**: smoke vs deep vs longhaul vs benchmark. Revisit markers whenever fixtures or
  algorithmic complexity changes.

## 5. Tooling considerations

- **Static analysis first**: prefer adding or tightening type hints and lint checks before writing a
  slow runtime assertion for the same contract.
- **Coverage and mutation testing**: consider enabling coverage reports or mutation testing when we
  suspect blind spots. Introduce these as optional tooling until the signal justifies CI integration.
- **Benchmark storage**: keep autosave paths stable and ensure any new benchmark suite integrates with
  existing Make targets.
- **Profiling recipes**: limit to hot kernels with known performance risks. Document how to reproduce
  the profile locally and how to interpret the artefacts.
- **Documentation hooks**: significant new testing patterns should be reflected in onboarding docs so
  agents follow the same playbook.

## 6. Decision checklist

Before landing a new or modified test case, confirm that:

- [ ] The guarded risk and quality dimension are clearly stated.
- [ ] Existing guards are insufficient or have been strengthened instead.
- [ ] Runtime tier placement matches expected wall-time and frequency.
- [ ] Fixtures and parameters are the minimal deterministic set that still exposes the bug class.
- [ ] Assertions provide actionable diagnostics with justified tolerances.
- [ ] Maintenance and artefact plans are documented (including escalation triggers).
- [ ] Documentation or task briefs reference the new expectations so future agents stay aligned.

Following this checklist keeps the validation stack coherent, enforceable, and explainable to new
contributors.

## 7. Operational workflow and tooling conventions

To keep quality control predictable across the next months, apply the following conventions in
addition to the decision criteria above.

- **Golden-path commands**: Before opening a PR or handing off work, run `just precommit`
  (format/lint/typecheck/full smoke). Use `just precommit-fast` + `just test-incremental` during
  local iteration, and `just test-deep` before merging performance-sensitive changes. Schedule
  `just test-longhaul` for regression sweeps, and the matching benchmark targets
  (`just bench`, `just bench-deep`, `just bench-longhaul`) when validating runtime budgets.
- **Pytest markers**: Tag new tests with `@pytest.mark.smoke`, `@pytest.mark.deep`, or
  `@pytest.mark.longhaul` explicitly when their runtime requires it. Leave smoke-only tests unmarked
  if they satisfy the enforced budget (10 s per-test via per-item markers, 60 s session cap). Keep
  benchmark modules under `tests/performance/` and mark them with both `smoke`/`deep` tiers when
  appropriate plus `@pytest.mark.benchmark`.
- **CI alignment**: `just ci` mirrors the GitHub Actions pipeline (format → lint → typecheck → smoke
  tests). If a change needs additional tooling (coverage, mutation testing, profiling), wire it in
  via a `just` recipe first and mirror the invocation in CI only once the cost and signal are proven.
- **Type-first guarding**: When adding runtime tests, audit whether stricter type hints or Ruff rules
  would catch the issue earlier. Prefer tightening `pyright` coverage before introducing slow smoke
  cases.
- **Artefact hygiene**: Store benchmark outputs under `.benchmarks/` and profiling traces under
  `.profiles/`. Commit baselines only when policy allows (invariant JSON under `tests/_baselines/` with
  maintainer approval). Capture any manual runs in the relevant task brief (§12 status section).
- **Timeout policy**: Apply `pytest` `timeout` markers or fixtures to smoke-tier tests that risk
  exceeding per-test budgets. Implement Make-level guards if suite runtime drifts towards the CI
  ceiling. Keep the pytest-testmon cache warm by running `just test-incremental` frequently and a
  full `just test` whenever dependencies shift significantly.
- **Re-tiering cadence**: Revisit marker assignments whenever runtime measurements change by more than
  ~20% or new fixtures land. Document adjustments in `docs/tasks/...` so subsequent agents inherit the
  rationale.
- **Escalation triggers**: Open a `Needs-Unblock` note if smoke-tier runtime exceeds 5 minutes,
  benchmarks regress by >10% without explanation, or the validation stack requires new infrastructure
  (coverage tooling, additional deps) not yet provisioned.

These conventions ensure contributors can rely on a consistent validation workflow while iterating on
both correctness and performance features.
