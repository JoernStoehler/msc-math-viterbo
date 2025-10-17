#!/usr/bin/env bash
set -euo pipefail

# dev-status — Print status for owner services inside the devcontainer.

echo "== In-container service status =="
echo "VibeKanban UI:"; (pgrep -a vibe-kanban || pgrep -fa 'npx .*vibe-kanban' || true)
echo
echo "VS Code Tunnel:"; (pgrep -a code || true)
echo
echo "cloudflared:"; (pgrep -a cloudflared || true)
echo
exit 0

