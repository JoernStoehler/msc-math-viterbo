#!/usr/bin/env bash
set -euo pipefail

# post-start — clean, opinionated bootstrap for the project owner workflow.
# - Keep shell ergonomics sane
# - Ensure permissions for bind-mounted dirs
# - Fast idempotent uv sync
# - Print actionable hints (auth status), but do not auto-start services
# - All service control lives in .devcontainer/Justfile

echo "[post-start] Booting environment (owner workflow)."

# Persist bash history to a mounted volume
HIST_DIR="$HOME/.bash_history_dir"
HIST_FILE="$HIST_DIR/.bash_history"
mkdir -p "$HIST_DIR" && touch "$HIST_FILE"
export HISTFILE="$HIST_FILE" || true

touch "$HOME/.bashrc" "$HOME/.bash_profile"
if ! grep -qE 'HISTFILE=.*\\.bash_history_dir/.bash_history' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ''
    echo '# Persist bash history to mounted volume'
    echo 'export HISTFILE="$HOME/.bash_history_dir/.bash_history"'
  } >> "$HOME/.bashrc"
fi
if ! grep -q '\\.bashrc' "$HOME/.bash_profile" 2>/dev/null; then
  echo 'test -f ~/.bashrc && . ~/.bashrc' >> "$HOME/.bash_profile"
fi

# Ensure PATH precedence for user-local bin
mkdir -p "$HOME/.local/bin"
if ! grep -q '\.local/bin' "$HOME/.bashrc" 2>/dev/null; then
  {
    echo ''
    echo '# Ensure user-local bin precedes system'
    echo 'export PATH="$HOME/.local/bin:$PATH"'
  } >> "$HOME/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"

# Fix permissions on bind-mounted folders (idempotent)
echo "[post-start] Verifying ownership for mounted folders (idempotent)."
for d in \
  "$HOME/.config/gh" \
  "$HOME/.vscode" \
  "$HOME/.config/codex" \
  "$HOME/.cloudflared" \
  "$HOME/.cache/uv" \
  "$HOME/.local/share/ai/bloop/vibe-kanban" \
  "/var/tmp/vibe-kanban/worktrees" \
; do
  [ -d "$d" ] || continue
  chown -R "$USER:$USER" "$d" || true
done

# Idempotent uv sync (fast when nothing changed)
if [ -f "pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  echo "[post-start] uv sync (dev extras, idempotent)."
  uv sync --extra dev >/dev/null || true
fi

# Diagnostics (non-fatal but actionable)
echo "[post-start] Diagnostics:"

# VS Code CLI
if command -v code >/dev/null 2>&1; then
  if code --help 2>/dev/null | grep -q "tunnel"; then
    echo "  - VS Code CLI: ok (tunnel supported)."
  else
    echo "  - VS Code CLI: present, but tunnel subcommand missing."
  fi
else
  echo "  - VS Code CLI: not found (will limit tunnel usage)."
fi

# GitHub CLI
if command -v gh >/dev/null 2>&1; then
  if gh auth status -h github.com >/dev/null 2>&1; then
    echo "  - gh: authenticated."
  else
    echo "  - gh: not authenticated. Run 'gh auth login'."
  fi
else
  echo "  - gh: not found."
fi

# Cloudflared status
CF_TUNNEL="${CF_TUNNEL:-vibekanban}"
if command -v cloudflared >/dev/null 2>&1; then
  if [ -s "$HOME/.cloudflared/cert.pem" ]; then
    if cloudflared tunnel list 2>/dev/null | grep -qE "^\s*${CF_TUNNEL}\b"; then
      echo "  - cloudflared: logged in; tunnel '${CF_TUNNEL}' defined."
    else
      echo "  - cloudflared: logged in; tunnel '${CF_TUNNEL}' not found. Create with: cloudflared tunnel create ${CF_TUNNEL}"
    fi
  else
    echo "  - cloudflared: not logged in. Run 'cloudflared tunnel login'."
  fi
else
  echo "  - cloudflared: not found."
fi

# Wrangler (Cloudflare) status
if command -v wrangler >/dev/null 2>&1; then
  echo "  - wrangler: installed. Use 'just -f .devcontainer/Justfile cf-worker-deploy' to deploy the font worker."
else
  echo "  - wrangler: not found. Install with 'npm i -g wrangler' if you plan to deploy the font worker."
fi

# Mount presence (informational)
for d in \
  "$HOME/.config/gh" \
  "$HOME/.vscode" \
  "$HOME/.config/codex" \
  "$HOME/.cloudflared" \
  "$HOME/.cache/uv" \
  "$HOME/.local/share/ai/bloop/vibe-kanban" \
  "/var/tmp/vibe-kanban/worktrees" \
; do
  [ -d "$d" ] || echo "  - mount missing (expected directory): $d"
done

echo "[post-start] Done. Use '.devcontainer/Justfile' to start services:"
echo "  - just -f .devcontainer/Justfile start-vibe"
echo "  - just -f .devcontainer/Justfile start-tunnel"
echo "  - just -f .devcontainer/Justfile start-cf"
