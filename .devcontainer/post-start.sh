#!/usr/bin/env bash
set -euo pipefail

echo "[post-start] Container started."

# Persist bash history to mounted volume
HIST_DIR="$HOME/.bash_history_dir"
HIST_FILE="$HIST_DIR/.bash_history"
mkdir -p "$HIST_DIR" && touch "$HIST_FILE"
export HISTFILE="$HIST_FILE" || true

if [ ! -f "$HOME/.bashrc" ]; then touch "$HOME/.bashrc"; fi
if ! grep -qE 'HISTFILE=.*\.bash_history_dir/.bash_history' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ''
    echo '# Persist bash history to mounted volume'
    echo 'export HISTFILE="$HOME/.bash_history_dir/.bash_history"'
  } >> "$HOME/.bashrc"
fi

if [ ! -f "$HOME/.bash_profile" ]; then touch "$HOME/.bash_profile"; fi
if ! grep -q '\.bashrc' "$HOME/.bash_profile" 2>/dev/null; then
  echo 'test -f ~/.bashrc && . ~/.bashrc' >> "$HOME/.bash_profile"
fi

# Make the project ready: instantiate and precompile (fast on subsequent boots)
if [ -f "Project.toml" ]; then
  "$HOME/.julia/juliaup/bin/julia" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' || true
fi

echo "[post-start] History and project ready."
