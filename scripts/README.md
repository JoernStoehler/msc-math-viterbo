# Scripts (stub)

During migration, DevOps scripts are intentionally minimal. The CI calls `just ci`, which:
- syncs dependencies with uv
- runs Ruff lint
- runs Pyright (basic)
- runs smoke-tier tests

`inc_select.py` powers the incremental pytest loop. Highlights:

- Run via `uv run --script scripts/inc_select.py [options]` or implicitly through `just test`.
- Writes impacted test file paths (and previously failing nodeids) to stdout; the Justfile consumes the result with pytest's `@argsfile` feature.
- Accepts runtime tuning:
  - `--plumbing PATH_SUFFIX` (repeatable) to add project-specific triggers that invalidate the incremental cache.
  - `--debug` to echo the change/selection sets for troubleshooting.
- Baselines the module graph in `.cache/inc_graph.json`; the selector updates the snapshot whenever it exits successfully.

Forward options from Just by exporting `INC_ARGS`, e.g. `INC_ARGS="--debug" just test`.

Legacy utilities (e.g., test metadata reporters) were removed from the default Justfile flow. Reintroduce them only if they provide clear value on the new stack.

