#!/usr/bin/env bash
set -euo pipefail

# owner-status-host — Non-destructive host diagnostics for devcontainer-related services.

echo "== Processes (host) =="
for patt in 'code .*tunnel' 'cloudflared' 'vibe-kanban' 'npx .*vibe-kanban' 'devcontainer'; do
  echo "-- $patt"
  pgrep -fa "$patt" || true
  echo
done

echo "== Cloudflared config dirs =="
for d in "$HOME/.cloudflared" "/srv/devhome/.cloudflared"; do
  if [ -d "$d" ]; then
    echo "-- $d"; ls -la "$d" || true; echo
  fi
done

echo "== Listening TCP ports (best-effort) =="
if command -v ss >/dev/null 2>&1; then
  ss -lntp 2>/dev/null || true
elif command -v netstat >/dev/null 2>&1; then
  netstat -lntp 2>/dev/null || true
else
  echo "No ss/netstat available."
fi

echo
echo "Hints: use 'bash .devcontainer/bin/owner-down.sh' to stop container/services."
