#!/usr/bin/env bash
set -euo pipefail

# owner-up — Host-orchestrated startup for devcontainer + services.
#
# Actions:
#   1) Ensure devcontainer CLI is available on the host
#   2) Start (or reuse) the devcontainer for this workspace
#   3) Run preflight checks inside the container
#   4) Start VS Code Tunnel, Cloudflared, and VibeKanban (detached)
#   5) Verify final status
#
# Idempotency:
# - Safe to re-run. Existing processes are left untouched; missing ones are started.
# - Fails early with actionable messages without stopping running services.

log() { printf '[owner-up] %s\n' "$*"; }
fail() { printf '[owner-up] ERROR: %s\n' "$*" >&2; exit 1; }

# Resolve workspace folder (defaults to repo root)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

# Guard: do not run inside the devcontainer
if [ -f "/.dockerenv" ] || [ -n "${DEVCONTAINER:-}" ] || [ -d "/workspaces" ]; then
  fail "Run this on the HOST, not inside the devcontainer."
fi

# Host prerequisites
command -v devcontainer >/dev/null 2>&1 || fail $'devcontainer CLI not found.\n  - Install via: bash .devcontainer/bin/host-install.sh\n  - Or see: https://github.com/devcontainers/cli'

# Optional: warn if similar services already run on the host
if pgrep -fa 'code .*tunnel' >/dev/null 2>&1; then
  log "Host VS Code tunnel detected; continuing (container will start its own)."
fi
if pgrep -fa 'cloudflared' >/dev/null 2>&1; then
  log "Host cloudflared detected; continuing (container runs isolated config)."
fi

log "Starting/ensuring devcontainer for: ${WORKSPACE_DIR}"
devcontainer up --workspace-folder "${WORKSPACE_DIR}" >/dev/null

run_in_container() {
  devcontainer exec --workspace-folder "${WORKSPACE_DIR}" bash -lc "$*"
}

log "Preflight checks (inside container)"
run_in_container "bash .devcontainer/bin/dev-install.sh --preflight" || {
  cat >&2 <<EOF
[owner-up] Preflight failed. Fix the reported issues and re-run.
  - To configure Cloudflared and DNS once: bash .devcontainer/bin/owner-cloudflare-setup.sh
  - To install host prerequisites:       bash .devcontainer/bin/host-install.sh
EOF
  exit 1
}

log "Starting services (detached) inside container"
run_in_container "bash .devcontainer/bin/dev-start.sh --detached"

# Give processes a moment to come up
sleep 2

log "Status after start"
run_in_container "bash .devcontainer/bin/dev-status.sh || true"

log "Post-check complete (see status above)."

log "All set. Attach to the session if you want logs:"
echo "  devcontainer exec --workspace-folder \"${WORKSPACE_DIR}\" bash -lc 'tmux attach -t viterbo-owner'"
