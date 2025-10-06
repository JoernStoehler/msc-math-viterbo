#!/usr/bin/env bash
# Run only changed tests vs. base (default origin/main). Falls back if mapping fails.
set -euo pipefail
BASE=${BASE:-origin/main}
ARGS=("$@")
if git rev-parse --verify "$BASE" >/dev/null 2>&1; then
  changed_py=$(git diff --name-only --diff-filter=AMR "$BASE"... | grep -E '\\.py$' || true)
else
  changed_py=$(git status --porcelain=v1 --untracked-files=normal | cut -c4- | grep -E '\\.py$' || true)
fi
changed_tests=$(echo "$changed_py" | grep -E '^tests/.*_test\\.py$' || true)
if [[ -n "$changed_tests" ]]; then
  uv run pytest "${ARGS[@]}" $changed_tests
  exit 0
fi
changed_src=$(echo "$changed_py" | grep -E '^src/.*\\.py$' || true)
if [[ -n "$changed_src" ]]; then
  tests=()
  while read -r f; do
    base=$(basename "$f" .py)
    cand="tests/test_${base}.py"
    [[ -f "$cand" ]] && tests+=("$cand")
  done <<< "$changed_src"
  if [[ ${#tests[@]} -gt 0 ]]; then
    uv run pytest "${ARGS[@]}" "${tests[@]}"
    exit 0
  fi
fi
uv run pytest "${ARGS[@]}" -m "smoke" -k "not slow and not gpu and not integration and not jit"
