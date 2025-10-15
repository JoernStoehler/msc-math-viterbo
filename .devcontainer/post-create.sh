#!/usr/bin/env bash
set -euo pipefail

# Purpose: make the devcontainer ready for daily Python development.
# 1. Ensure modern Python tooling (uv, pip) is available.
# 2. Install project dependencies in editable mode.
# 3. Install aux tools (ripgrep, Codex CLI) that help collaborators.

echo "[post-create] Preparing Python tooling stack"

PYTHON=${PYTHON:-python3}

"$PYTHON" -m pip install --upgrade pip >/dev/null

# Install uv (fast Python package manager) and bail if it cannot be installed.
if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] Installing uv (https://github.com/astral-sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
fi
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] ERROR: uv not available after installation attempt." >&2
  echo "[post-create] The project assumes uv is present; rerun the installer or install manually." >&2
  exit 1
fi

echo "[post-create] Syncing project dependencies with uv (lockfile-driven)"
uv sync --extra dev >/dev/null

SUDO=""
if command -v sudo >/dev/null 2>&1; then
  SUDO=sudo
fi

# Install tmux and terminfo (best-effort) because workflows rely on them.
RG_VERSION="14.1.0"

echo "[post-create] Ensuring ripgrep, tmux, and terminfo are installed"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y >/dev/null || true
  if ! command -v tmux >/dev/null 2>&1; then $SUDO apt-get install -y tmux >/dev/null || true; fi
  # Install extra terminfo entries (tmux-256color) for proper color support
  $SUDO apt-get install -y ncurses-term >/dev/null || true
  # Install Poppler utils for PDF -> text/markdown extraction (pdftotext)
  if ! command -v pdftotext >/dev/null 2>&1; then $SUDO apt-get install -y poppler-utils >/dev/null || true; fi
fi

if command -v rg >/dev/null 2>&1 && rg --version | head -n1 | grep -q "rg ${RG_VERSION}"; then
  echo "[post-create] ripgrep ${RG_VERSION} already installed"
else
  echo "[post-create] Installing ripgrep ${RG_VERSION} (x86_64 binary)"
  archive="ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl.tar.gz"
  url="https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/${archive}"
  tmp_dir=$(mktemp -d)
  if curl -fsSL "$url" -o "${tmp_dir}/${archive}" && \
     tar -xzf "${tmp_dir}/${archive}" -C "$tmp_dir" --strip-components=1 "ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl/rg"; then
    mkdir -p "${HOME}/.local/bin"
    install -m 755 "${tmp_dir}/rg" "${HOME}/.local/bin/rg" && \
      echo "[post-create] Installed ripgrep ${RG_VERSION}"
  else
    echo "[post-create] WARNING: Failed to download or extract ripgrep ${RG_VERSION}" >&2
  fi
  rm -rf "$tmp_dir"
fi

# Install just (best-effort) so the canonical task runner is available inside the container.
echo "[post-create] Ensuring just is installed"
if ! command -v just >/dev/null 2>&1; then
  echo "[post-create] Installing just via official installer script"
  mkdir -p "${HOME}/.local/bin"
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to "${HOME}/.local/bin" >/dev/null 2>&1 || true
fi
if ! command -v just >/dev/null 2>&1; then
  echo "[post-create] ERROR: just not available after installation attempt; install manually and rerun." >&2
  exit 1
fi

# Install the Codex CLI when npm is present.
if command -v npm >/dev/null 2>&1; then
  echo "[post-create] Installing Codex CLI (@openai/codex)"
  npm i -g @openai/codex >/dev/null 2>&1 || true
fi

echo "[post-create] Ensuring VS Code CLI (with 'tunnel') is installed"

install_vscode_cli_simple(){
  # Single path: download CLI tarball from Microsoft CDN and install 'code'.
  local arch; arch="$(uname -m)"
  local os_flavor="cli-linux-x64"
  if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then os_flavor="cli-linux-arm64"; fi
  if command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q "tunnel"; then
    echo "[post-create] VS Code CLI already present"
    return 0
  fi
  mkdir -p "${HOME}/.local/bin"
  export PATH="$HOME/.local/bin:$PATH"
  local url="https://update.code.visualstudio.com/latest/${os_flavor}/stable"
  local tmpdir
  tmpdir="$(mktemp -d)"
  if curl -fsSLo "${tmpdir}/cli.tar.gz" "$url"; then
    if tar -xzf "${tmpdir}/cli.tar.gz" -C "$tmpdir" code; then
      install -m 0755 "${tmpdir}/code" "${HOME}/.local/bin/code"
    fi
  fi
  rm -rf "$tmpdir"
  if command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q "tunnel"; then
    echo "[post-create] Installed VS Code CLI"
  else
    echo "[post-create] ERROR: VS Code CLI not available; required for tunnels." >&2
    exit 1
  fi
}

install_vscode_cli_simple

echo "[post-create] Post-create steps complete"
