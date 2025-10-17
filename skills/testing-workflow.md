---
name: testing-workflow
description: Run and interpret the project’s testing and linting suite efficiently using shared just/uv commands.
last-updated: 2025-10-17
---

# Testing Workflow

## Default Loop

1. Format, lint, type-check, and run smoke tests via `just checks`.
2. If formatting or linting fails, run `just fix`; rerun `just checks` afterwards.
3. For focused smoke tests, run `just test`. Pass incremental selector options with `INC_ARGS="..." just test`.
4. Before opening a PR or after substantial refactors, run `just ci` for the full parity workflow.

## Incremental Selector

- `scripts/inc_select.py` computes impacted tests; invoked automatically by `just test`.
- Use `INC_ARGS="--debug"` to inspect selection decisions when troubleshooting.
- Keep `.cache/inc_graph.json` under version control as produced; do not reset it manually mid-task.

## Benchmarking

- Run `just bench` for smoke-tier benchmarks when performance regressions are suspected.
- Bench outputs land under `.benchmarks/`; review changes before committing.

## Troubleshooting

- Prefer `uv run` to execute scripts (`uv run python -m pytest ...`) to ensure dependencies match `uv.lock`.
- If tests mutate temporary directories, ensure paths stay within the workspace to respect sandboxing.
- Capture failing command output in the task notes; maintainers expect exact command strings and revisions tested.

## Related Skills

- `coding-standards` — aligns code style and architecture before tests.
- `repo-onboarding` — ensures prerequisite steps completed before running test suites.
