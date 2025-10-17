#!/usr/bin/env bash
set -euo pipefail

# owner-rebuild — Rebuild and restart the devcontainer for this workspace.
# Safe to run when the container is not up. Attempts a graceful stop first.

log() { printf '[owner-rebuild] %s\n' "$*"; }
warn() { printf '[owner-rebuild] WARNING: %s\n' "$*" >&2; }
fail() { printf '[owner-rebuild] ERROR: %s\n' "$*" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

# Guard: do not run inside the devcontainer
if [ -f "/.dockerenv" ] || [ -n "${DEVCONTAINER:-}" ] || [ -d "/workspaces" ]; then
  fail "Run this on the HOST, not inside the devcontainer."
fi

# Flags
NO_CACHE=0
for arg in "$@"; do
  case "$arg" in
    --no-cache) NO_CACHE=1 ;;
  esac
done

command -v devcontainer >/dev/null 2>&1 || fail $'devcontainer CLI not found.\n  - Install via: bash .devcontainer/bin/host-install.sh\n  - Or see: https://github.com/devcontainers/cli'

run_in_container() {
  devcontainer exec --workspace-folder "${WORKSPACE_DIR}" bash -lc "$*"
}

# Try to stop services if the container is running
if run_in_container "echo __OK__" >/dev/null 2>&1; then
  log "Devcontainer is running; stopping services before rebuild."
  run_in_container ".devcontainer/bin/dev-stop.sh" || warn "Failed to stop services (continuing)."
else
  log "No running devcontainer detected; proceeding with rebuild."
fi

log "Rebuilding devcontainer (remove existing container)"
if [ "$NO_CACHE" -eq 1 ]; then
  devcontainer up --workspace-folder "${WORKSPACE_DIR}" --remove-existing-container --build-no-cache
else
  devcontainer up --workspace-folder "${WORKSPACE_DIR}" --remove-existing-container
fi

log "Running preflight checks inside the container"
if ! run_in_container "bash .devcontainer/bin/dev-install.sh --preflight"; then
  warn "Preflight reported issues; see the output above."
fi

log "Starting services inside the container (detached)"
run_in_container "bash .devcontainer/bin/dev-start.sh --detached"
log "Status after start"
run_in_container "bash .devcontainer/bin/dev-status.sh || true"

log "Rebuild complete."
