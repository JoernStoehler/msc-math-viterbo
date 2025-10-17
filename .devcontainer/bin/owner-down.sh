#!/usr/bin/env bash
set -euo pipefail

# owner-down — Host-orchestrated shutdown for devcontainer + services.
#
# Actions:
#   1) Ensure devcontainer CLI is available on the host
#   2) Best-effort stop services inside the container (dev-stop)
#   3) Stop the devcontainer for this workspace
#   4) Print a brief post-check for stray host processes
#
# Idempotency:
# - Safe to re-run. If the container is not running, owner-stop/exec is skipped.
# - Does not kill host processes automatically; only reports.

log() { printf '[owner-down] %s\n' "$*"; }
warn() { printf '[owner-down] WARNING: %s\n' "$*" >&2; }
fail() { printf '[owner-down] ERROR: %s\n' "$*" >&2; exit 1; }

# Resolve workspace folder (defaults to repo root)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

# Guard: do not run inside the devcontainer
if [ -f "/.dockerenv" ] || [ -n "${DEVCONTAINER:-}" ] || [ -d "/workspaces" ]; then
  fail "Run this on the HOST, not inside the devcontainer."
fi

# Host prerequisites
command -v devcontainer >/dev/null 2>&1 || {
  cat >&2 <<EOF
[owner-down] ERROR: devcontainer CLI not found.
  - Install via: just -f .devcontainer/Justfile host-install-devcontainer
  - Or see: https://github.com/devcontainers/cli
EOF
  exit 1
}

run_in_container() {
  devcontainer exec --workspace-folder "${WORKSPACE_DIR}" bash -lc "$*"
}

# Detect CLI capabilities (down support varies by version)
if devcontainer --help | grep -qE '^\s*devcontainer down\b'; then
  HAS_DOWN=1
else
  HAS_DOWN=0
fi

# Probe if the workspace devcontainer is running
IS_RUNNING=0
if run_in_container "echo __OK__" >/dev/null 2>&1; then
  IS_RUNNING=1
fi

if [ "$IS_RUNNING" -eq 1 ]; then
  log "Devcontainer appears to be RUNNING for: ${WORKSPACE_DIR}"
  log "Stopping services inside container (best-effort)"
  if ! run_in_container ".devcontainer/bin/dev-stop.sh"; then
    warn "owner-stop failed inside container (continuing shutdown)."
  fi

  if [ "$HAS_DOWN" -eq 1 ]; then
    log "Stopping devcontainer (CLI supports 'down')"
    if ! devcontainer down --workspace-folder "${WORKSPACE_DIR}" >/dev/null 2>&1; then
      warn "'devcontainer down' reported an error; container may require manual stop via Docker/Podman."
    fi
  else
    warn "Your devcontainer CLI does not support 'down'. Skipping programmatic stop."
    warn "If the container persists, stop it via your container runtime (e.g., 'docker stop <name>')."
  fi
else
  log "No running devcontainer detected for this workspace. Nothing to stop."
  if [ "$HAS_DOWN" -eq 0 ]; then
    log "Note: devcontainer CLI lacks 'down'; this is normal on some versions."
  fi
fi

log "Post-check on host for stray processes"
for patt in 'code .*tunnel' 'cloudflared' 'vibe-kanban'; do
  if pgrep -fa "$patt" >/dev/null 2>&1; then
    echo "  - Found: $patt";
    pgrep -fa "$patt" || true;
  fi
done

log "Done. If anything above remains and should be stopped, kill those PIDs explicitly."

log "Summary: CLI has 'down'=$HAS_DOWN, devcontainer running=$IS_RUNNING"
