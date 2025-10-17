#!/usr/bin/env bash
set -euo pipefail

# dev-stop — Best-effort stop of owner services inside the devcontainer.

log() { printf '[dev-stop] %s\n' "$*"; }
warn() { printf '[dev-stop] WARNING: %s\n' "$*" >&2; }
fail() { printf '[dev-stop] ERROR: %s\n' "$*" >&2; exit 1; }

# Must run inside the devcontainer
if [ -z "${LOCAL_DEVCONTAINER:-}" ] && [ ! -f "/.dockerenv" ] && [ ! -d "/workspaces" ]; then
  fail "Run this INSIDE the devcontainer."
fi

log "Stopping VibeKanban / Tunnel / Cloudflared (best-effort)."

# VibeKanban
if pgrep -fa 'npx .*vibe-kanban' >/dev/null 2>&1 || pgrep -fa '(^|/)vibe-kanban( |$)' >/dev/null 2>&1; then
  log "  - vibe-kanban: running; stopping"
  pgrep -fa 'npx .*vibe-kanban' || true
  pgrep -fa '(^|/)vibe-kanban( |$)' || true
  pkill -f 'npx .*vibe-kanban' || true
  pkill -f '(^|/)vibe-kanban( |$)' || true
  sleep 0.2
  if pgrep -fa 'npx .*vibe-kanban' >/dev/null 2>&1 || pgrep -fa '(^|/)vibe-kanban( |$)' >/dev/null 2>&1; then
    warn "  - vibe-kanban: still running (manual kill may be needed)"
  else
    log  "  - vibe-kanban: stopped"
  fi
else
  log "  - vibe-kanban: not running"
fi

# VS Code Tunnel
if pgrep -fa 'code .*tunnel' >/dev/null 2>&1; then
  log "  - code tunnel: running; stopping"
  pgrep -fa 'code .*tunnel' || true
  pkill -f 'code .*tunnel' || true
  sleep 0.2
  if pgrep -fa 'code .*tunnel' >/dev/null 2>&1; then
    warn "  - code tunnel: still running (manual kill may be needed)"
  else
    log  "  - code tunnel: stopped"
  fi
else
  log "  - code tunnel: not running"
fi

# Cloudflared
if pgrep -fa 'cloudflared .*tunnel' >/dev/null 2>&1 || pgrep -fa '(^|/)cloudflared( |$)' >/dev/null 2>&1; then
  log "  - cloudflared: running; stopping"
  pgrep -fa 'cloudflared .*tunnel' || true
  pgrep -fa '(^|/)cloudflared( |$)' || true
  pkill -f 'cloudflared .*tunnel' || true
  pkill -f '(^|/)cloudflared( |$)' || true
  sleep 0.2
  if pgrep -fa 'cloudflared .*tunnel' >/dev/null 2>&1 || pgrep -fa '(^|/)cloudflared( |$)' >/dev/null 2>&1; then
    warn "  - cloudflared: still running (manual kill may be needed)"
  else
    log  "  - cloudflared: stopped"
  fi
else
  log "  - cloudflared: not running"
fi

log "Stop sequence completed."

