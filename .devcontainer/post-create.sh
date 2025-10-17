#!/usr/bin/env bash
set -euo pipefail

# post-create — prepare the container for the owner workflow.
# Keep it simple and opinionated; install only what we need.

echo "[post-create] Preparing environment."

# Safety: refuse to run on host
if [ -z "${LOCAL_DEVCONTAINER:-}" ] && [ ! -f "/.dockerenv" ] && [ ! -d "/workspaces" ]; then
  echo "[post-create] ERROR: must be run inside the devcontainer. Use 'devcontainer up' on the host." >&2
  exit 1
fi

PYTHON=${PYTHON:-python3}
"$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || true

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "[post-create] Installing uv (https://github.com/astral-sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || {
    echo "[post-create] ERROR: failed to install uv." >&2
    return 1
  }
  export PATH="$HOME/.local/bin:$PATH"
}

ensure_misc_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  sudo apt-get install -y tmux ncurses-term >/dev/null 2>&1 || true
  command -v rg >/dev/null 2>&1 || sudo apt-get install -y ripgrep >/dev/null 2>&1 || true
  command -v pdftotext >/dev/null 2>&1 || sudo apt-get install -y poppler-utils >/dev/null 2>&1 || true
}

ensure_codex() {
  if command -v codex >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "[post-create] WARNING: npm not available; skip codex install." >&2
    return 0
  fi
  echo "[post-create] Installing codex"
  npm install -g @openai/codex >/dev/null 2>&1 || {
    echo "[post-create] WARNING: failed to install codex; install manually if needed." >&2
  }
}

ensure_uv || exit 1
bash .devcontainer/bin/dev-install.sh --install || exit 1
ensure_misc_packages
ensure_codex

if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] ERROR: uv still missing after installation attempt." >&2
  exit 1
fi

if [ -f "pyproject.toml" ]; then
  echo "[post-create] uv sync (dev extras)."
  uv sync --extra dev >/dev/null || true
fi

echo "[post-create] Environment ready. Use host 'owner-up.sh' or in-container 'dev-start.sh' to start services."
echo "[post-create] For Cloudflare DNS/config helpers see 'bash .devcontainer/bin/owner-cloudflare-setup.sh'."
