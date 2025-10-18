#!/usr/bin/env bash
set -euo pipefail

# owner-attach — Attach an interactive bash to the running devcontainer.
#
# Behavior:
# - Verifies devcontainer CLI is installed (host-side).
# - Errors if the devcontainer for this workspace is not running.
# - On success, replaces the current process with an interactive login shell in the container.

log() { printf '[owner-attach] %s\n' "$*"; }
fail() { printf '[owner-attach] ERROR: %s\n' "$*" >&2; exit 1; }

# Resolve workspace folder (defaults to repo root)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

# Guard: do not run inside the devcontainer
if [ -f "/.dockerenv" ] || [ -n "${DEVCONTAINER:-}" ] || [ -d "/workspaces" ]; then
  fail "Run this on the HOST, not inside the devcontainer."
fi

# Host prerequisite
command -v devcontainer >/dev/null 2>&1 || {
  cat >&2 <<EOF
[owner-attach] ERROR: devcontainer CLI not found.
  - Install via: bash .devcontainer/bin/host-install.sh
  - Docs: https://github.com/devcontainers/cli
EOF
  exit 1
}

run_in_container() {
  devcontainer exec --workspace-folder "${WORKSPACE_DIR}" bash -lc "$*"
}

# Probe if the devcontainer for this workspace is running
if ! run_in_container "echo __OK__" >/dev/null 2>&1; then
  fail $'No running devcontainer detected for this workspace.\n  - Start it first: bash .devcontainer/bin/owner-up.sh'
fi

log "Attaching interactive bash inside devcontainer for: ${WORKSPACE_DIR}"
# Replace current process with an interactive login shell in the container
exec devcontainer exec --workspace-folder "${WORKSPACE_DIR}" bash -l

