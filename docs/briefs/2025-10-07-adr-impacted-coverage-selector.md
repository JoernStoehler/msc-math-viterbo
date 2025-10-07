---
title: Impacted Tests via Coverage Contexts — ADR
date: 2025-10-07
status: adopted
owner: Core Eng
reviewers: Maintainer
tags: [testing, coverage, impacted, devops, xdist, jax]
---

Summary

- Adopt an additive “impacted tests” fast path that selects pytest nodeids from a coverage contexts
  map built on main/nightly. Fallback to the full suite when the selector cannot be trusted. Keep
  xdist optional due to potential JAX memory constraints.

Decision

- Adopt the coverage-contexts selector as the golden path for smoke-tier tests via `just test`
  (impacted by default, serial fallback to full).
- Keep the selector script `scripts/impacted_cov.py`. The `coverage` target refreshes the contexts
  map (`.cache/coverage.json`) inline, so no separate map-refresh step is required.
- Provide explicit full-run variants: `just test-full` (serial) and `just test-xdist` (parallel).
- Prefer serial impacted runs by default; enable xdist on demand.
- Maintain strict invalidation rules; prefer safety over under-testing.

Context

- Goal: preserve sub-minute to low-minute PR feedback as the suite grows, without depending on
  `pytest-testmon` and while retaining correctness.
- Constraints: JAX/JIT and xdist may increase memory use; coverage overhead is acceptable for
  scheduled map builds, not on every PR run.

Implementation Snapshot

- Selector script: `scripts/impacted_cov.py`
  - uv “script” header; no external deps; prints nodeids to stdout; returns `0` on selection, `2`
    for fallback.
  - Conservative invalidation: any change under `tests/`, any `conftest.py`, `pyproject.toml`,
    `pytest.ini`, `uv.lock`, `src/viterbo/__init__.py`, or the script itself forces full run.
  - Handles coverage contexts that contain a trailing `|run` suffix.
- Minimal coverage config: `pyproject.toml`

```toml
[tool.coverage.run]
branch = true
relative_files = true
source = ["src/viterbo"]

[tool.coverage.json]
show_contexts = true
```

- Justfile targets (at time of writing): `Justfile`

```make
test:
    @mkdir -p .cache
    @echo "Running smoke-tier pytest (impacted selection with fallback)."
    $UV run --script scripts/impacted_cov.py --base ${IMPACTED_BASE:-origin/main} --map .cache/coverage.json > .cache/impacted_nodeids.txt || true
    @if [ -s .cache/impacted_nodeids.txt ]; then \
        $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} @.cache/impacted_nodeids.txt {{PYTEST_ARGS}}; \
    else \
        $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} {{PYTEST_ARGS}}; \
    fi

test-full:
    @echo "Running full smoke-tier pytest (serial)."
    $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} {{PYTEST_ARGS}}

test-xdist:
    @echo "Running full smoke-tier pytest (-n auto)."
    $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} -n auto {{PYTEST_ARGS}}

coverage:
    @echo "Running smoke-tier tests with coverage (HTML + XML reports, serial) and refreshing contexts map."
    @mkdir -p .cache
    $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} --cov=src/viterbo --cov-context=test --cov-report=term-missing --cov-report=html --cov-report=xml --junitxml .cache/last-junit.xml {{PYTEST_ARGS}}
    $UV run coverage json -o .cache/coverage.json --show-contexts

impacted-xdist:
    @mkdir -p .cache
    @echo "Selecting impacted tests via coverage contexts (xdist)."
    $UV run --script scripts/impacted_cov.py --base ${IMPACTED_BASE:-origin/main} --map .cache/coverage.json > .cache/impacted_nodeids.txt || true
    @if [ -s .cache/impacted_nodeids.txt ]; then \
        echo "Running impacted tests (-n auto)"; \
        $UV run pytest -q -n auto @.cache/impacted_nodeids.txt {{PYTEST_ARGS}}; \
    else \
        echo "Fallback: running full test suite (-n auto)"; \
        $UV run pytest -q -n auto {{PYTEST_ARGS}}; \
    fi
```

Measurement (Pilot)

Warning: single-run measurements on a warm environment; use as directional guidance only. Repeat
with N≥5 for robust medians.

- Map build (serial, with contexts): ~105.8 s; JSON export: ~1.7 s
- Simulated tiny diff: touched one line in
  `src/viterbo/symplectic/capacity/facet_normals/subset_utils.py` to trigger impacted selection.
- Selector outcome: `impacted=21`, `total≈100`, `p≈0.21`, `selector≈120 ms`.

Commands Used

```bash
# 1) Refresh coverage reports + contexts JSON + last JUnit
just coverage

# 2) Simulate a tiny source change (edit a single line); later revert with git
$EDITOR src/viterbo/symplectic/capacity/facet_normals/subset_utils.py

# 3) Compute impacted nodeids (serial diff vs current HEAD), print metrics
uv run --script scripts/impacted_cov.py \
  --base HEAD --map .cache/coverage.json --verbose > .cache/impacted_nodeids.txt || true

# 4) Run impacted vs baselines (serial / xdist)
uv run pytest -q @.cache/impacted_nodeids.txt            # impacted serial (~32.5 s)
uv run pytest -q                                         # full serial (~74.6 s)
uv run pytest -q -n auto                                 # full xdist (~36.4 s)
uv run pytest -q -n auto @.cache/impacted_nodeids.txt    # impacted xdist (~25.6 s)

# Coverage timing (includes contexts map export)
just coverage

# Impacted with/without coverage timing
uv run pytest -q @.cache/impacted_nodeids.txt            # impacted (no coverage)
uv run pytest -q --cov=src/viterbo @.cache/impacted_nodeids.txt
uv run pytest -q --cov=src/viterbo --cov-context=test @.cache/impacted_nodeids.txt

Status-aware selection policy

- Inputs: latest contexts map `.cache/coverage.json` and last JUnit `.cache/last-junit.xml` (from
  `just coverage`).
- Run set = impacted_by_diff ∪ previously_failing ∪ new_tests (if detected).
- Skip set = previously_passing ∩ unaffected_by_diff.
- Threshold/fallback: if impacted fraction `p > 0.4`, treat as full run (no selection). Any
  invalidation (tests changed, fixtures, config, etc.) also forces full run.
- Messaging: the selector prints counts for rerun_impacted_pass, rerun_impacted_fail,
  rerun_prev_fail_unaffected, skip_unaffected_pass, unknown_impacted.

# 5) Optional FAST runs (JIT off)
FAST=1 JAX_DISABLE_JIT=true JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run pytest -q @.cache/impacted_nodeids.txt          # impacted FAST serial (~43.0 s)
FAST=1 JAX_DISABLE_JIT=true JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run pytest -q                                       # full FAST serial (~106.7 s)

# 6) Revert local experiment edits
git restore -- src/viterbo/symplectic/capacity/facet_normals/subset_utils.py
```

Observed Results (single run)

- Impacted serial: 32.5 s; full serial: 74.6 s (≈2.3× faster).
- Impacted xdist: 25.6 s; full xdist: 36.4 s (≈1.4× faster).
- Serial impacted (32.5 s) is faster than full xdist (36.4 s) in this run.
- Testmon serial: 97.5 s (slower than both serial full and impacted here).
- FAST (JIT off): impacted 43.0 s vs full 106.7 s (≈2.5× faster).

Selector Notes

- Coverage contexts often include a trailing `|run` suffix; the selector strips it before emitting
  nodeids.
- Path matching defaults to robust suffix matching (exact match available via `--strict-paths`).
- Invalidation rules bias to safety; any change under `tests/` forces full run.
- Threshold to force full run: `p > 0.4` by default (override with `IMPACTED_THRESHOLD`).

Operational Guidance

- Only build the coverage map on scheduled/nightly runs or on `main` pushes; PR runs consume the
  cached JSON and avoid coverage overhead.
- Prefer serial impacted runs when memory pressure is a concern with JAX and xdist.
- Print selector metrics in CI logs for visibility (impacted_count, total_in_map, p, selector_ms,
  fallback_reason).

Alternatives Considered (brief)

- Full serial (baseline): safest, slow on small diffs.
- `pytest-testmon` (serial): decent, but slower here and maintenance risk.
- Import-graph selector: simpler but less safe (misses dynamic links).
- Bazel/Pants: strong long-term, heavy adoption cost.

Future Work

- CI job to build/cache `.cache/coverage.json` on `main`/nightly, then consume in PRs.
- Add a small measurement harness to gather N≥5 replicate timings and print medians + MAD.
- Tune `p` threshold and invalidation list from experience; track fallback rate in CI.

Rollback Plan

- Delete `scripts/impacted_cov.py` and the four Justfile targets; remove coverage JSON cache from
  CI. Existing workflows remain unaffected.
