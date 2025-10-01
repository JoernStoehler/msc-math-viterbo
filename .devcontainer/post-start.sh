#!/usr/bin/env bash
set -euo pipefail

# Purpose: run lightweight tasks every time the devcontainer boots.
# 1. Persist bash history to a mounted volume.
# 2. Re-sync dependencies with uv to catch pyproject edits between boots.

echo "[post-start] Container started."

HIST_DIR="$HOME/.bash_history_dir"
HIST_FILE="$HIST_DIR/.bash_history"
mkdir -p "$HIST_DIR" && touch "$HIST_FILE"
export HISTFILE="$HIST_FILE" || true

if [ ! -f "$HOME/.bashrc" ]; then touch "$HOME/.bashrc"; fi
if ! grep -qE 'HISTFILE=.*\\.bash_history_dir/.bash_history' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ''
    echo '# Persist bash history to mounted volume'
    echo 'export HISTFILE="$HOME/.bash_history_dir/.bash_history"'
  } >> "$HOME/.bashrc"
fi

if [ ! -f "$HOME/.bash_profile" ]; then touch "$HOME/.bash_profile"; fi
if ! grep -q '\\.bashrc' "$HOME/.bash_profile" 2>/dev/null; then
  echo 'test -f ~/.bashrc && . ~/.bashrc' >> "$HOME/.bash_profile"
fi

if [ -f "pyproject.toml" ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "[post-start] Syncing dependencies with uv (idempotent)"
    uv pip install --system -e .[dev] >/dev/null || true
  else
    echo "[post-start] WARN: uv missing; run .devcontainer/post-create.sh" >&2
  fi
fi

echo "[post-start] Environment ready."
