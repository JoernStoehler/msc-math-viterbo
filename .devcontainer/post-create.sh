#!/usr/bin/env bash
set -euo pipefail

# post-create — prepare the container for the owner workflow.
# Keep it simple and opinionated; install only what we need.

echo "[post-create] Preparing environment."

PYTHON=${PYTHON:-python3}
"$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || true

# uv (fast Python toolchain)
if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] Installing uv (https://github.com/astral-sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[post-create] ERROR: uv not available." >&2; exit 1; }

# just (task runner)
if ! command -v just >/dev/null 2>&1; then
  echo "[post-create] Installing just"
  mkdir -p "${HOME}/.local/bin"
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to "${HOME}/.local/bin" >/dev/null 2>&1 || true
fi
command -v just >/dev/null 2>&1 || { echo "[post-create] ERROR: just not available." >&2; exit 1; }

# VS Code CLI (for 'tunnel')
echo "[post-create] Ensuring VS Code CLI is installed."
install_vscode_cli(){
  local arch os_flavor url tmpdir
  arch="$(uname -m)"
  os_flavor="cli-linux-x64"
  if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then os_flavor="cli-linux-arm64"; fi
  if command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q "tunnel"; then
    return 0
  fi
  mkdir -p "$HOME/.local/bin"
  url="https://update.code.visualstudio.com/latest/${os_flavor}/stable"
  tmpdir="$(mktemp -d)"
  if curl -fsSLo "${tmpdir}/cli.tar.gz" "$url" && tar -xzf "${tmpdir}/cli.tar.gz" -C "$tmpdir" code; then
    install -m 0755 "${tmpdir}/code" "${HOME}/.local/bin/code"
  fi
  rm -rf "$tmpdir"
}
install_vscode_cli
command -v code >/dev/null 2>&1 || { echo "[post-create] ERROR: VS Code CLI not available." >&2; exit 1; }

# Nice-to-haves (best-effort): ripgrep, tmux, ncurses-term, pdftotext
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -y >/dev/null 2>&1 || true
  sudo apt-get install -y tmux ncurses-term >/dev/null 2>&1 || true
  command -v rg >/dev/null 2>&1 || sudo apt-get install -y ripgrep >/dev/null 2>&1 || true
  command -v pdftotext >/dev/null 2>&1 || sudo apt-get install -y poppler-utils >/dev/null 2>&1 || true
fi

# Project deps (idempotent)
if [ -f "pyproject.toml" ]; then
  echo "[post-create] uv sync (dev extras)."
  uv sync --extra dev >/dev/null || true
fi

# Cloudflare Wrangler (optional but recommended for Worker deploys)
if command -v npm >/dev/null 2>&1; then
  if ! command -v wrangler >/dev/null 2>&1; then
    echo "[post-create] Installing Cloudflare wrangler CLI"
    npm i -g wrangler >/dev/null 2>&1 || true
  fi
fi

echo "[post-create] Environment ready. Use '.devcontainer/Justfile' to start services."
