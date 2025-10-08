#!/usr/bin/env bash
set -euo pipefail

# Purpose: make the devcontainer ready for daily Python development.
# 1. Ensure modern Python tooling (uv, pip) is available.
# 2. Install project dependencies in editable mode.
# 3. Install aux tools (ripgrep, Codex CLI) that help collaborators.

echo "[post-create] Preparing Python tooling stack"

PYTHON=${PYTHON:-python3}

"$PYTHON" -m pip install --upgrade pip >/dev/null

# Install uv (fast Python package manager) and bail if it cannot be installed.
if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] Installing uv (https://github.com/astral-sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
fi
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] ERROR: uv not available after installation attempt." >&2
  echo "[post-create] The project assumes uv is present; rerun the installer or install manually." >&2
  exit 1
fi

echo "[post-create] Syncing project dependencies with uv (lockfile-driven)"
uv sync --extra dev >/dev/null

# Install ripgrep, tmux, and terminfo (best-effort) because workflows rely on them.
echo "[post-create] Ensuring ripgrep, tmux, and terminfo are installed"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
  $SUDO apt-get update -y >/dev/null || true
  if ! command -v rg >/dev/null 2>&1; then $SUDO apt-get install -y ripgrep >/dev/null || true; fi
  if ! command -v tmux >/dev/null 2>&1; then $SUDO apt-get install -y tmux >/dev/null || true; fi
  # Install extra terminfo entries (tmux-256color) for proper color support
  $SUDO apt-get install -y ncurses-term >/dev/null || true
fi

# Install just (best-effort) so the canonical task runner is available inside the container.
echo "[post-create] Ensuring just is installed"
if ! command -v just >/dev/null 2>&1; then
  echo "[post-create] Installing just via official installer script"
  mkdir -p "${HOME}/.local/bin"
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to "${HOME}/.local/bin" >/dev/null 2>&1 || true
fi
if ! command -v just >/dev/null 2>&1; then
  echo "[post-create] ERROR: just not available after installation attempt; install manually and rerun." >&2
  exit 1
fi

# Install the Codex CLI if Node/npm is present so collaborators can use familiar tooling.
if command -v npm >/dev/null 2>&1; then
  echo "[post-create] Installing Codex CLI (@openai/codex)"
  npm i -g @openai/codex >/dev/null 2>&1 || true
fi

echo "[post-create] Post-create steps complete"
