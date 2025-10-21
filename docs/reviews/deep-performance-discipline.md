Status: Implemented (scope: performance discipline deep-dive; caveats: reflects repository state on 2025-10-20)

# Review — Performance Discipline Deep-Dive

Provenance
- Topic: Performance Discipline Deep-Dive
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 87fdebc
- Timebox: ~90 minutes
- Scope: tests/performance, pytest markers/timeout, Justfile benches, runtime time-budget guard, math hot paths (polytope, capacity_ehz stubs), skills (performance-discipline, good-code-loop), mkdocs build
- Version: v0.1

Context
- Goal: Deep-dive on performance posture: benchmark coverage, smoke/deep/longhaul tiers, measurement strategy, time budgets (runtime.enforce_time_budget), thresholds for C++ escalation, and profiling guidance. Output is an unfiltered, unsorted list of findings plus commands/outcomes.

Inputs & Method
- Commands run (representative):
  - `uv run python scripts/load_skills_metadata.py --quiet`
  - `rg -n "enforce_time_budget|TimeBudgetExceededError|benchmark|bench|profil|smoke|deep|longhaul|timeout"`
  - `sed -n '1,220p' <file>` targeted reads (kept ≤250 lines per read)
  - `git rev-parse --short HEAD`
  - `uv run mkdocs build --strict` (post-write)
- Reviewed: `tests/performance/*`, `pytest.ini`, `Justfile` bench/profile targets, `src/viterbo/runtime.py`, `src/viterbo/math/polytope.py`, `src/viterbo/math/capacity_ehz/stubs.py` (oriented-edge), skills `performance-discipline` and `good-code-loop`, docs architecture overview.

Findings (unsorted)
- Benchmark coverage exists but is thin: two smoke-tier benches in `tests/performance/` — `test_benchmark_math.py` (support on a 2k×16 cloud) and `test_cpp_extension_benchmark.py` (affine C++ vs Torch baseline). Good starting point but no benches for H↔V conversions, oriented-edge spectrum, or EHZ multipliers.
- Benchmarks are correctly gated: `pytest-benchmark` missing → skip; C++ ext missing → skip. Keeps smoke tier green on minimal setups; reinforces portability.
- Autosave and storage are configured: `Justfile` sets `--benchmark-only --benchmark-autosave --benchmark-storage=.benchmarks`. Results persist for local comparisons; not wired to CI by design.
- Tiering is consistent: pytest markers define `smoke`, `deep`, `longhaul` in `pytest.ini`; `Justfile` provides `bench`, `bench-deep`, `bench-longhaul`. Only smoke benches currently exist; deep/longhaul benches can be added when scenarios stabilize.
- Smoke tests cover perf-adjacent paths: `tests/test_cpp_extension.py` ensures extension plumbing with Python fallbacks; `tests/test_smoke.py` hits polytope queries. However, no time-budget assertion tests (e.g., expecting `TimeBudgetExceededError` under extreme inputs) — potential guard gap.
- Measurement strategy aligns with skills: `skills/performance-discipline.md` prescribes “measure first, profile after, escalate to C++ with evidence”; current benches and placeholders reflect that intent. RNG is pinned in benches (`torch.manual_seed(0)`), good for determinism.
- Environment capture is partial: benches don’t record device/dtype or environment snapshots beyond pytest-benchmark’s defaults. Consider adding a small helper to print `torch.__version__`, `dtype`, device, and CPU info into the benchmark group name or extra columns for easier across-machine comparisons.
- PyTest timeout posture: global `timeout = 120` in `pytest.ini` gives a hard cap per test. `SMOKE_TEST_TIMEOUT` exists in `Justfile` but is not wired into pytest; could remove or connect via `--timeout` to avoid drift.
- Runtime time budgets are implemented at the function level via `@enforce_time_budget()` (defaulting to env `VITERBO_SOLVER_TIMEOUT`, 30s fallback). Signal path uses `ITIMER_REAL` with proper handler restore; non-signal platforms detect overrun post-execution — clear behavior trade-off.
- Only `oriented_edge_spectrum_4d` currently uses `@enforce_time_budget()`; other potentially heavy routes (e.g., future EHZ QP/LP stubs, large H↔V conversions) don’t. Intentional minimalism, but adding opt-in decorators (disabled by default) could standardize guardrails.
- Budget semantics are sensible: decorator argument allows per-call override; env-based default makes CI reproducible. Non-positive seconds disables, which is practical for profiling runs.
- Oriented-edge pruning controls are rich: rotation-number cap, optional Chaidez–Hutchings per-face budgets (`use_cF_budgets`, certified constant builder), and dominance memoization keyed by quantized transfer; these are performance levers but lack targeted micro-benches to validate effect sizes and variance.
- Suggested bench additions (smoke or deep):
  - `polytope.vertices_to_halfspaces` (ConvexHull-backed) and `halfspaces_to_vertices` on controlled 3D/4D shapes; exercise scale with N-vertices and facet counts; assert runtime stays within budget tier envelopes.
  - Oriented-edge spectrum on a tiny, deterministic 4D product (existing fixtures) with tight caps; measure sensitivity to `rotation_cap`, `use_cF_budgets`, and memo toggles.
  - EHZ Haim–Kislev candidate enumeration path (already used in `capacity_ehz_haim_kislev`) with small supports; benchmark candidate generation and DP max triangle stage separately.
- Benchmark assertions: consider adding light guardrails (not strict pass/fail on ns deltas) such as “median runtime < X ms on CI CPU baseline” in smoke or deep tiers. Use wide tolerances and skip on variability; document acceptance in skills.
- Regression tracking: pytest-benchmark supports compare; adopting a simple historical JSON/CSV summary under `artefacts/` for weekly snapshots would ease trend detection without bloating the repo.
- C++ escalation policy is clear in skills and architecture docs: maintain Python fallbacks; escalate only for sustained hotspots. No codified numeric threshold yet; propose a working bar: “≥3× speedup on median with ≤10% variance across 10 runs, same outputs, same dtype/device; maintenance cost justified.”
- C++ examples (`add_one`, `affine_scale_shift`) are intentionally trivial; they validate toolchain and path. Next C++ candidate should be profiling-led (e.g., inner loop in oriented-edge or heavy linear algebra fallback when Torch lacks an op); avoid premature CUDA.
- Profiling guidance is partially implemented: `Justfile` `profile`/`profile-line` are placeholders; a notebook `notebooks/atlas_tiny_profile.py` provides cProfile helpers focused on dataset building (useful pattern). Consider adding a scripted callgrind wrapper and basic flamegraph guidance under `skills/performance-discipline.md` or wiring the `profile` target.
- Torch profiler is not referenced; for GPU work in models it could be added later. For math (CPU-bound), prefer `py-spy`, `scalene`, or `cProfile` + snakeviz; keep artefacts out of Git, under `.profiles/` per `Justfile`.
- Device/dtype policy supports performance hygiene: math defaults to float64 and respects caller device; C++ fallbacks maintain dtype/device. Benchmarks should explicitly set dtype/device to avoid incidental slowdowns (e.g., float64 BLAS on some CPUs).
- CI strategy: `just ci` runs mkdocs build and smoke-tier tests but not benches. Correct trade-off for speed; deep performance validation remains manual/pre-merge via `bench(-deep)`.
- Docs and reviews already reference time budgets positively (e.g., Review 04); this deep-dive complements them by proposing concrete bench coverage and thresholds.
- Potential edge: H↔V conversions rely on SciPy’s `ConvexHull` on CPU; environments without SciPy wheels may degrade install velocity. Keep the Python path but note the dependency in bench docs; allow skip if SciPy unavailable.
- Naming: benchmarks use plain test names; grouping with `@pytest.mark.parametrize("size", [...])` and `benchmark.group = ...` could improve readability in reports.
- Warmup/variance: pytest-benchmark defaults may need upping reps on noisy machines; document a convention (e.g., `--benchmark-min-rounds=5` locally when comparing) but don’t bake it into CI.
- Time budget vs test timeout: clarify precedence for contributors — test-level timeout aborts the whole test; function-level budget raises `TimeBudgetExceededError`. For algorithms, prefer the latter plus a try/except in tests to assert budget behavior without killing the session.
- Documentation: add a short “Performance Playbook” subsection under `skills/performance-discipline.md` enumerating the above acceptance bar, bench folders, and how to read `.benchmarks/` results; keep PR evidence expectations explicit in `good-code-loop`.

Actions (first pass)
- Add smoke benches for `vertices_to_halfspaces` and `halfspaces_to_vertices` (3D/4D small fixtures; deterministic).
- Add a deep bench for `oriented_edge_spectrum_4d` with tight caps and toggles (`rotation_cap`, `use_cF_budgets`, `use_memo`).
- Add a micro-bench for EHZ candidate enumeration + DP triangle step; start with k≤6.
- Wire `Justfile profile` to a simple callgrind wrapper or `py-spy` record into `.profiles/` (kept out of Git).
- Consider a tiny smoke test asserting `TimeBudgetExceededError` on a pathological oriented-edge input when `VITERBO_SOLVER_TIMEOUT` is set very small (and skip on non-signal platforms if flaky).
- Decide and document a provisional C++ escalation threshold; require evidence (bench tables + profiles) in tickets.
- Either remove `SMOKE_TEST_TIMEOUT` from `Justfile` or plumb it into pytest via `--timeout` to avoid drift.

Validation
- Skills metadata: `uv run python scripts/load_skills_metadata.py --quiet` — OK
- Docs build: `uv run mkdocs build --strict` — OK

Status
- Finalized
