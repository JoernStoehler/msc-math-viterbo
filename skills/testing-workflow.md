---
name: testing-workflow
description: Run and interpret the project’s testing and linting suite efficiently using shared just/uv commands.
last-updated: 2025-10-17
---

# Testing Workflow

## Default Loop

1. Format, lint, type-check, and run smoke tests via `just checks`.
   - Invokes Ruff (lint + format), Pyright (basic), and Pytest smoke tier.
2. If formatting or linting fails, run `just fix`; rerun `just checks` afterwards to confirm clean state.
3. For focused smoke tests, run `just test`. Pass incremental selector options with `INC_ARGS="..." just test`.
4. Before opening a PR or after substantial refactors, run `just ci` for the full parity workflow.

## Incremental Selector

- `scripts/inc_select.py` computes impacted tests; invoked automatically by `just test`.
- Use `INC_ARGS="--debug"` to inspect selection decisions when troubleshooting.
- Keep `.cache/inc_graph.json` under version control as produced; do not reset it manually mid-task.

## Static Analysis Notes

- Pyright runs in basic mode; address type errors promptly or justify suppressions in task notes.
- Ruff handles import ordering (`I` rules) and selected bugbear/pyupgrade checks. If automatic fixes are available, `just fix` will apply them.
- `just lint` also runs `scripts/load_skills_metadata.py` (output suppressed) to validate skill frontmatter; fix any warnings before rerunning.
- Avoid ignoring lint/type errors via `# noqa` or `type: ignore` unless policy requires and you document reasoning.

## Benchmarking

- Run `just bench` for smoke-tier benchmarks when performance regressions are suspected.
- Bench outputs land under `.benchmarks/`; review changes before committing.

## CI Parity

- `just ci` mirrors the GitHub Actions workflow; run it before handoff when changes affect core modules, infrastructure, or cross-cutting behavior.
- Record runtime and notable failures in task notes; attach log excerpts if CI parity reveals flakiness.

## Troubleshooting

- Prefer `uv run` to execute scripts (`uv run python -m pytest ...`) to ensure dependencies match `uv.lock`.
- If tests mutate temporary directories, ensure paths stay within the workspace to respect sandboxing.
- Capture failing command output in the task notes; maintainers expect exact command strings and revisions tested.
- When diagnosing intermittent failures, rerun the relevant subset with `pytest -k "<pattern>" --maxfail=1 --disable-warnings`.
- Escalate when a regression persists after two targeted attempts or touches shared infrastructure (CI workflow, devcontainer, base deps).

## Related Skills

- `coding-standards` — aligns code style and architecture before tests.
- `repo-onboarding` — ensures prerequisite steps completed before running test suites.
- `performance-discipline` — guides benchmarking and profiling efforts that follow testing.
