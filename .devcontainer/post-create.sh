#!/usr/bin/env bash
set -euo pipefail

# post-create — prepare the container for the owner workflow.
# Keep it simple and opinionated; install only what we need.

echo "[post-create] Preparing environment."

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

ensure_just() {
  if command -v just >/dev/null 2>&1; then
    return 0
  fi
  echo "[post-create] Installing just"
  mkdir -p "$HOME/.local/bin"
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --to "$HOME/.local/bin" >/dev/null 2>&1 || {
      echo "[post-create] ERROR: failed to install just." >&2
      return 1
    }
}

ensure_vscode_cli() {
  if command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q "tunnel"; then
    return 0
  fi
  echo "[post-create] Installing VS Code CLI"
  mkdir -p "$HOME/.local/bin"
  local arch=os url tmpdir
  arch="$(uname -m)"
  os="cli-linux-x64"
  if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
    os="cli-linux-arm64"
  fi
  url="https://update.code.visualstudio.com/latest/${os}/stable"
  tmpdir="$(mktemp -d)"
  if curl -fsSLo "${tmpdir}/cli.tar.gz" "$url" \
    && tar -xzf "${tmpdir}/cli.tar.gz" -C "$tmpdir" code; then
    install -m 0755 "${tmpdir}/code" "$HOME/.local/bin/code"
  else
    echo "[post-create] ERROR: failed to install VS Code CLI." >&2
    rm -rf "$tmpdir"
    return 1
  fi
  rm -rf "$tmpdir"
}

ensure_cloudflared() {
  if command -v cloudflared >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[post-create] ERROR: apt-get unavailable; cannot install cloudflared automatically." >&2
    return 1
  fi
  echo "[post-create] Installing cloudflared via Cloudflare APT repo"
  local key=/usr/share/keyrings/cloudflare-main.gpg
  local list=/etc/apt/sources.list.d/cloudflared.list
  sudo mkdir -p --mode=0755 "$(dirname "$key")"
  if [ ! -f "$key" ]; then
    curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
      | sudo tee "$key" >/dev/null || {
        echo "[post-create] ERROR: unable to fetch Cloudflare APT key." >&2
        return 1
      }
  fi
  if ! sudo test -f "$list" || ! sudo grep -q "pkg.cloudflare.com/cloudflared" "$list"; then
    echo "deb [signed-by=${key}] https://pkg.cloudflare.com/cloudflared focal main" \
      | sudo tee "$list" >/dev/null
  fi
  sudo apt-get update -y >/dev/null 2>&1 || {
    echo "[post-create] ERROR: apt-get update failed." >&2
    return 1
  }
  sudo apt-get install -y cloudflared >/dev/null 2>&1 || {
    echo "[post-create] ERROR: apt-get install cloudflared failed." >&2
    return 1
  }
}

ensure_wrangler() {
  if command -v wrangler >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "[post-create] WARNING: npm not available; skip wrangler install." >&2
    return 0
  fi
  echo "[post-create] Installing Cloudflare wrangler CLI"
  npm i -g wrangler >/dev/null 2>&1 || {
    echo "[post-create] WARNING: failed to install wrangler; install manually if needed." >&2
  }
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
ensure_just || exit 1
ensure_vscode_cli || exit 1
ensure_cloudflared || exit 1
ensure_wrangler
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

echo "[post-create] Environment ready. Use '.devcontainer/Justfile' to start services."
echo "[post-create] For Cloudflare DNS/config helpers see 'just -f .devcontainer/Justfile owner-cloudflare-*'."
