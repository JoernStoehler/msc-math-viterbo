#!/usr/bin/env bash
set -euo pipefail

echo "[post-create] Installing Julia via Juliaup (official)"

# Keep it simple: one cache at ~/.julia, ensure ownership, install via juliaup with -p and --add-to-path
JULIA_DIR="$HOME/.julia"
JULIAUP_HOME="$JULIA_DIR/juliaup"

mkdir -p "$JULIA_DIR"
chown -R "$(id -u):$(id -g)" "$JULIA_DIR" || true

# Do not pre-create $JULIAUP_HOME; let the installer create it. Do not delete caches automatically.
curl -fsSL https://install.julialang.org | sh -s -- -y --add-to-path=true -p "$JULIAUP_HOME" || true

if [ -x "$JULIAUP_HOME/bin/juliaup" ]; then
  "$JULIAUP_HOME/bin/juliaup" add 1.11 || true
  "$JULIAUP_HOME/bin/juliaup" default 1.11 || true
else
  echo "[post-create] ERROR: juliaup not found at $JULIAUP_HOME/bin/juliaup." >&2
  echo "[post-create] Hints: ensure $JULIA_DIR is writable (chown once), and that $JULIAUP_HOME does not already exist as a non-install." >&2
  echo "[post-create] If a stale $JULIAUP_HOME exists from a failed run, remove it once and rebuild." >&2
  exit 1
fi

echo "[post-create] Julia installed and on PATH."

# Make julia and juliaup available via a common PATH directory without rc or container env tweaks
mkdir -p "$HOME/.local/bin"
ln -sf "$JULIAUP_HOME/bin/julia" "$HOME/.local/bin/julia"
ln -sf "$JULIAUP_HOME/bin/juliaup" "$HOME/.local/bin/juliaup"
echo "[post-create] Symlinked julia and juliaup into $HOME/.local/bin"

# Minimal developer ergonomics: install ripgrep (best-effort)
# Keep it simple and fast; skip if already present. Use sudo if available.
if ! command -v rg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[post-create] Installing ripgrep via apt"
    export DEBIAN_FRONTEND=noninteractive
    if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=""; fi
    $SUDO apt-get update -y || true
    $SUDO apt-get install -y ripgrep || true
    if ! command -v rg >/dev/null 2>&1; then
      echo "[post-create] WARN: ripgrep not found after apt install; may require new shell or manual install" >&2
    fi
  fi
fi

# Codex CLI (best-effort if Node/npm present)
if command -v npm >/dev/null 2>&1; then
  echo "[post-create] Installing Codex CLI (@openai/codex)"
  npm i -g @openai/codex >/dev/null 2>&1 || true
fi
