#!/usr/bin/env bash
set -euo pipefail

# host-install — Host-side helpers to prepare the workstation for the devcontainer workflow.

log() { printf '[host-install] %s\n' "$*"; }
warn() { printf '[host-install] WARNING: %s\n' "$*" >&2; }
fail() { printf '[host-install] ERROR: %s\n' "$*" >&2; exit 1; }

# Must run on host (not inside container)
if [ -f "/.dockerenv" ] || [ -n "${DEVCONTAINER:-}" ] || [ -d "/workspaces" ]; then
  fail "Run this on the HOST, not inside the devcontainer."
fi

ensure_devcontainer_cli() {
  if command -v devcontainer >/dev/null 2>&1; then
    log "devcontainer CLI already installed ($(devcontainer --version 2>/dev/null || echo unknown))."
    return 0
  fi
  if command -v npm >/dev/null 2>&1; then
    log "Installing devcontainer CLI via npm (@devcontainers/cli)"
    npm i -g @devcontainers/cli || fail "npm install failed"
    devcontainer --version || true
  else
    fail "npm not found; install Node.js/npm or follow https://github.com/devcontainers/cli"
  fi
}

ensure_host_dirs() {
  local dirs=(
    /srv/devhome/.config/gh
    /srv/devhome/.config/.wrangler
    /srv/devhome/.vscode
    /srv/devhome/.config/codex
    /srv/devhome/.cloudflared
    /srv/devhome/.cache/uv
    /srv/devhome/.local/share/ai/bloop/vibe-kanban
    /srv/devworktrees/vibe-kanban/worktrees
  )
  log "Ensuring host bind-mount roots exist (sudo may prompt)..."
  sudo mkdir -p "${dirs[@]}"
  local owner="${SUDO_USER:-${USER}}"
  sudo chown -R "$owner:$owner" /srv/devhome /srv/devworktrees
  log "Host directories ready under /srv/devhome and /srv/devworktrees."
}

ensure_devcontainer_cli
ensure_host_dirs

log "Host ready. Next: bash .devcontainer/bin/owner-up.sh"

