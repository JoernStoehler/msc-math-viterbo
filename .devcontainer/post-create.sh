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
bash .devcontainer/bin/container-admin install || exit 1
ensure_misc_packages
ensure_codex

if ! command -v uv >/dev/null 2>&1; then
  echo "[post-create] ERROR: uv still missing after installation attempt." >&2
  exit 1
fi

pick_torch_index() {
  # Allow explicit override
  if [ -n "${TORCH_CUDA_CHANNEL:-}" ]; then
    case "$TORCH_CUDA_CHANNEL" in
      cu124) echo "https://download.pytorch.org/whl/cu124"; return;;
      cu121) echo "https://download.pytorch.org/whl/cu121"; return;;
      cpu)   echo "https://download.pytorch.org/whl/cpu"; return;;
    esac
  fi
  # Auto-detect via nvidia-smi; fall back to CPU
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver_line=$(nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: [0-9]+\.[0-9]+' || true)
    if [ -n "$ver_line" ]; then
      cuda_ver=${ver_line##*: } # e.g. 12.4
      major=${cuda_ver%%.*}
      minor=${cuda_ver##*.}
      if [ "$major" = "12" ] && [ "$minor" -ge 4 ] 2>/dev/null; then
        echo "https://download.pytorch.org/whl/cu124"; return
      fi
      if [ "$major" = "12" ] && [ "$minor" -ge 1 ] 2>/dev/null; then
        echo "https://download.pytorch.org/whl/cu121"; return
      fi
    fi
  fi
  echo "https://download.pytorch.org/whl/cpu"
}

if [ -f "pyproject.toml" ]; then
  TORCH_INDEX_URL=$(pick_torch_index)
  echo "[post-create] Installing torch (index: ${TORCH_INDEX_URL}) and dev deps (idempotent)."
  uv pip install --system \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "https://pypi.org/simple" \
    "torch==2.5.1" >/dev/null || true
  uv pip install --system \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "https://pypi.org/simple" \
    -e ".[dev]" >/dev/null || true
fi

echo "[post-create] Environment ready."
echo "[post-create] Start services:"
echo "  - Host: bash .devcontainer/bin/host-admin up preflight start --interactive"
echo "  - In container: bash .devcontainer/bin/container-admin start --detached"
echo "[post-create] Cloudflared DNS/config helper: bash .devcontainer/bin/container-admin cf-setup"
