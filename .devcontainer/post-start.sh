#!/usr/bin/env bash
set -euo pipefail

# post-start — clean, opinionated bootstrap for the project owner workflow.
# - Keep shell ergonomics sane
# - Ensure permissions for bind-mounted dirs
# - Fast idempotent uv sync
# - Print actionable hints (auth status); does not auto-start services
# - Service control lives in .devcontainer/bin/* scripts (no Justfile)

echo "[post-start] Booting environment (owner workflow)."

# Safety: refuse to run on host
if [ -z "${LOCAL_DEVCONTAINER:-}" ] && [ ! -f "/.dockerenv" ] && [ ! -d "/workspaces" ]; then
  echo "[post-start] ERROR: must be run inside the devcontainer. Use 'devcontainer up' on the host." >&2
  exit 1
fi

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
  "$HOME/.config/.wrangler" \
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
CF_HOSTNAME="${CF_HOSTNAME:-vibekanban.joernstoehler.com}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
if command -v cloudflared >/dev/null 2>&1; then
  if [ -s "$HOME/.cloudflared/cert.pem" ]; then
    if cloudflared tunnel info "${CF_TUNNEL}" >/dev/null 2>&1; then
      echo "  - cloudflared: logged in; tunnel '${CF_TUNNEL}' defined."

      CF_CONFIG="${CLOUDFLARED_CONFIG:-$HOME/.cloudflared/config-${CF_TUNNEL}.yml}"
      if [ -f "$CF_CONFIG" ]; then
        echo "  - cloudflared: config $(basename "$CF_CONFIG") present."
      else
        echo "  - cloudflared: config $(basename "$CF_CONFIG") missing; run 'bash .devcontainer/bin/owner-cloudflare-setup.sh'."
      fi
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
  WRANGLER_DIR=".devcontainer/cloudflare"
  WRANGLER_TOML="${WRANGLER_DIR}/wrangler.toml"
  if [ -f "$WRANGLER_TOML" ]; then
    WORKER_NAME="$(grep -E '^name\s*=' "$WRANGLER_TOML" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
  fi
  WORKER_NAME="${WORKER_NAME:-vk-font}"
  if [ -d "$WRANGLER_DIR" ]; then
    if DEPLOYMENTS_JSON=$(cd "$WRANGLER_DIR" && wrangler deployments list --name "$WORKER_NAME" --json 2>/dev/null); then
      if printf '%s' "$DEPLOYMENTS_JSON" | grep -q '"created_on"'; then
        echo "  - wrangler: worker '${WORKER_NAME}' deployed."
      elif printf '%s' "$DEPLOYMENTS_JSON" | grep -Eq '^\s*\[\s*\]\s*$'; then
        echo "  - wrangler: authenticated; worker '${WORKER_NAME}' has no deployments yet. Run 'cd .devcontainer/cloudflare && wrangler deploy'."
      else
        echo "  - wrangler: installed; deployment status unknown (check 'wrangler deployments list')."
      fi
    else
      if [ -n "${CLOUDFLARE_API_TOKEN:-}" ] || [ -n "${CLOUDFLARE_API_KEY:-}" ]; then
        echo "  - wrangler: installed; failed to query deployments (see 'wrangler deployments list')."
      else
        echo "  - wrangler: installed (set CLOUDFLARE_API_TOKEN to check deployment status)."
      fi
    fi
  else
    echo "  - wrangler: installed."
  fi
else
  echo "  - wrangler: not found. Install with 'npm i -g wrangler' if you plan to deploy the font worker."
fi

# Mount presence (informational)
for d in \
  "$HOME/.config/gh" \
  "$HOME/.config/.wrangler" \
  "$HOME/.vscode" \
  "$HOME/.codex" \
  "$HOME/.cloudflared" \
  "$HOME/.cache/uv" \
  "$HOME/.local/share/ai/bloop/vibe-kanban" \
  "/var/tmp/vibe-kanban/worktrees" \
; do
  [ -d "$d" ] || echo "  - mount missing (expected directory): $d"
done

echo "[post-start] Services are not auto-started. To start manually:"
echo "  - In container: bash .devcontainer/bin/container-admin start --detached"
echo "  - Or on host:   bash .devcontainer/bin/host-admin up preflight start --interactive"
