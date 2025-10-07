---
title: Incremental Tests via Module Import Graph — ADR
date: 2025-10-07
status: adopted
owner: Core Eng
reviewers: Maintainer
tags: [testing, incremental, imports, devops, xdist, jax]
---

Summary

- Adopt a simple, Git‑agnostic incremental selector based on Python module imports and file hashes.
- Run only tests in changed/new test files and tests that import changed modules; always include
  previously failing tests. Skip entirely when there are no Python changes and no prior failures.
- Fall back to a full smoke run on risky (plumbing) or large changes.

Decision

- Golden path: `just test` uses `scripts/inc_select.py` to select tests; it prints test file paths
  to run plus nodeids of previously failing tests, or a skip message on “no changes/no failures”.
- Explicit full variants: `just test-full` (serial) and `just test-xdist` (parallel).
- Maintain strict invalidation (plumbing) and large‑impact fallback.

Implementation Snapshot

- Selector: `scripts/inc_select.py`

  - AST‑based import graph over `**/*.py` (internal modules only); compute file hashes.
  - Dirty = added/removed/modified files; propagate to importers (reverse edges) to discover dirty
    test files; also mark changed/new test files dirty.
  - Always include last failing nodeids (parsed from `.cache/last-junit.xml`, written by pytest).
  - Skip‑on‑no‑diff: if no Python changes and no prior failures, write `.cache/impacted_none` and
    print a clear message; `just test` does not invoke pytest in this case.
  - Guardrails: changes to conftest.py, pytest.ini, pytest config in `pyproject.toml`, `uv.lock`, or
    this script advise a full run.

- Justfile targets (excerpt)

```make
test:
    @mkdir -p .cache
    @echo "Running smoke-tier pytest (incremental selection with fallback)."
    @rm -f .cache/impacted_none
    $UV run --script scripts/inc_select.py > .cache/impacted_nodeids.txt || true
    @if [ -s .cache/impacted_nodeids.txt ]; then \
        $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} --junitxml .cache/last-junit.xml @.cache/impacted_nodeids.txt {{PYTEST_ARGS}}; \
    elif [ -f .cache/impacted_none ]; then \
        echo "Selector: no changes and no prior failures — skipping pytest run."; \
    else \
        $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} --junitxml .cache/last-junit.xml {{PYTEST_ARGS}}; \
    fi

coverage:
    @echo "Running smoke-tier tests with coverage (HTML + XML reports, serial)."
    @mkdir -p .cache
    $UV run pytest -q {{PYTEST_SMOKE_FLAGS}} --cov=src/viterbo --cov-report=term-missing --cov-report=html --cov-report=xml --junitxml .cache/last-junit.xml {{PYTEST_ARGS}}
```

Selection Policy

- Inputs: static import graph + file hashes + last JUnit. No coverage contexts required for
  selection (coverage remains for reports).
- Run = tests in changed/new test files ∪ tests that (transitively) import changed modules ∪ last
  failing nodeids.
- Skip = tests unaffected by the change set. If no Python changes and no prior failures, skip pytest
  entirely.
- Fallbacks: plumbing changes or large impact → full smoke run.

Notes & Caveats

- Dynamic imports and heavy monkey‑patching can evade static import graphs; when detected in changed
  modules, advise a full run.
- Import cycles reduce efficacy (more tests selected).
- CI policy: local runs use incremental selection; CI (`just ci`) runs the full smoke tier.

Alternatives Considered

- Coverage‑contexts selector: precise but adds overhead and complexity; not needed for module‑level
  selection goals.
- `pytest-incremental`: similar import‑graph approach; lacks our skip‑on‑no‑diff, explicit
  guardrails, CI/local split, and xdist policy.

Rollback Plan

- Delete `scripts/inc_select.py` and revert `just test` to a full smoke run. Existing workflows
  remain unaffected.
