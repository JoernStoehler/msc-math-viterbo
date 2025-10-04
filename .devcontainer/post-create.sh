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
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] ERROR: uv not available after installation attempt." >&2
  echo "[post-create] The project assumes uv is present; rerun the installer or install manually." >&2
  exit 1
fi

echo "[post-create] Syncing project dependencies with uv (lockfile-driven)"
uv sync --extra dev >/dev/null

# Install ripgrep (best-effort) because many workflows rely on it for search.
echo "[post-create] Ensuring ripgrep is installed"
if ! command -v rg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
  $SUDO apt-get update -y >/dev/null || true
  $SUDO apt-get install -y ripgrep >/dev/null || true
fi

# Install the Codex CLI if Node/npm is present so collaborators can use familiar tooling.
if command -v npm >/dev/null 2>&1; then
  echo "[post-create] Installing Codex CLI (@openai/codex)"
  npm i -g @openai/codex >/dev/null 2>&1 || true
fi

echo "[post-create] Post-create steps complete"
