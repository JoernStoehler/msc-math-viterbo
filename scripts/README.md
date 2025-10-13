# Scripts (stub)

During migration, DevOps scripts are intentionally minimal. The CI calls `just ci`, which:
- syncs dependencies with uv
- runs Ruff lint
- runs Pyright (basic)
- runs smoke-tier tests

Legacy utilities (e.g., test metadata reporters) were removed from the default Justfile flow. Reintroduce them only if they provide clear value on the new stack.

