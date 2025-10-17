#!/usr/bin/env bash
set -euo pipefail

# dev-start — Start owner services inside the devcontainer (idempotent).
# Services: VibeKanban UI (npx), VS Code Tunnel, Cloudflared named tunnel.

log() { printf '[dev-start] %s\n' "$*"; }
warn() { printf '[dev-start] WARNING: %s\n' "$*" >&2; }
fail() { printf '[dev-start] ERROR: %s\n' "$*" >&2; exit 1; }

# Must run inside the devcontainer
if [ -z "${LOCAL_DEVCONTAINER:-}" ] && [ ! -f "/.dockerenv" ] && [ ! -d "/workspaces" ]; then
  fail "Run this INSIDE the devcontainer."
fi

FRONTEND_PORT=${FRONTEND_PORT:-3000}
HOST_BIND=${HOST:-0.0.0.0}
TUNNEL_NAME=${TUNNEL_NAME:-viterbo-dev}
CF_TUNNEL=${CF_TUNNEL:-vibekanban}

# Resolve repo root for commands that expect to run there (e.g., code tunnel)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

DETACHED=1
for arg in "$@"; do
  case "$arg" in
    --foreground) DETACHED=0 ;;
    --detached)   DETACHED=1 ;;
  esac
done

log "Preparing to start services (detached=${DETACHED})."
mkdir -p "$HOME/.logs"

# Preflight (soft): print actionable hints but don't abort unless essential
if ! command -v npx >/dev/null 2>&1; then
  warn "npx not found; VibeKanban cannot start. Install Node.js."
fi
if command -v code >/dev/null 2>&1; then
  if ! code --help 2>/dev/null | grep -q "tunnel"; then
    warn "VS Code CLI present but 'tunnel' subcommand missing; skipping tunnel."
  fi
else
  warn "VS Code CLI not found; skipping tunnel."
fi
if ! command -v cloudflared >/dev/null 2>&1; then
  warn "cloudflared not found; skipping Cloudflare tunnel."
fi

# Ensure tmux session if available
SESSION="viterbo-owner"
USE_TMUX=0
if command -v tmux >/dev/null 2>&1 && [ "$DETACHED" -eq 1 ]; then
  USE_TMUX=1
  tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n admin 'bash -lc :'
fi

start_cmd_detached() {
  local name="$1"; shift
  local cmd="$*"
  if [ "$USE_TMUX" -eq 1 ]; then
    tmux new-window -t "$SESSION" -n "$name" "bash -lc '$cmd'" >/dev/null
    log "  - $name: started (tmux window '$name')"
  else
    nohup bash -lc "$cmd" >"$HOME/.logs/${name}.log" 2>&1 < /dev/null & disown || true
    log "  - $name: started (nohup, logs in ~/.logs/${name}.log)"
  fi
}

# VibeKanban UI
if pgrep -fa 'npx .*vibe-kanban' >/dev/null 2>&1 || pgrep -fa '(^|/)vibe-kanban( |$)' >/dev/null 2>&1; then
  log "  - vibe-kanban: already running"
elif command -v npx >/dev/null 2>&1; then
  CMD="HOST='${HOST_BIND}' PORT='${FRONTEND_PORT}' FRONTEND_PORT='${FRONTEND_PORT}' npx --yes vibe-kanban"
  if [ "$DETACHED" -eq 1 ]; then
    start_cmd_detached "vibe" "$CMD"
  else
    log "  - vibe-kanban: starting in foreground"
    eval "$CMD"
  fi
else
  warn "  - vibe-kanban: skipped (npx missing)"
fi

# VS Code Tunnel
if pgrep -fa 'code .*tunnel' >/dev/null 2>&1; then
  log "  - code tunnel: already running"
elif command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q "tunnel"; then
  CMD="cd '${root_dir}' && code tunnel --accept-server-license-terms --name '${TUNNEL_NAME}'"
  if [ "$DETACHED" -eq 1 ]; then
    start_cmd_detached "tunnel" "$CMD"
  else
    log "  - code tunnel: starting in foreground"
    eval "$CMD"
  fi
else
  warn "  - code tunnel: skipped (CLI missing or no 'tunnel' subcommand)"
fi

# Cloudflared named tunnel
CF_CONFIG_DEFAULT="$HOME/.cloudflared/config-${CF_TUNNEL}.yml"
CF_CONFIG_PATH="${CLOUDFLARED_CONFIG:-$CF_CONFIG_DEFAULT}"
if pgrep -fa 'cloudflared .*tunnel' >/dev/null 2>&1 || pgrep -fa '(^|/)cloudflared( |$)' >/dev/null 2>&1; then
  log "  - cloudflared: already running"
elif command -v cloudflared >/dev/null 2>&1; then
  if [ ! -f "$CF_CONFIG_PATH" ]; then
    warn "  - cloudflared: config missing at $CF_CONFIG_PATH; skipping"
  else
    CMD="cloudflared tunnel --config '$CF_CONFIG_PATH' run '${CF_TUNNEL}'"
    if [ "$DETACHED" -eq 1 ]; then
      start_cmd_detached "cloudflared" "$CMD"
    else
      log "  - cloudflared: starting in foreground"
      eval "$CMD"
    fi
  fi
else
  warn "  - cloudflared: skipped (binary missing)"
fi

log "Start sequence complete."
