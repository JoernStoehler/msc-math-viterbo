#!/usr/bin/env bash
set -euo pipefail

# dev-install — Install and/or check tools used by owner services (inside container).
# Handles: VS Code CLI, cloudflared, wrangler. Does not install Node itself.

log() { printf '[dev-install] %s\n' "$*"; }
warn() { printf '[dev-install] WARNING: %s\n' "$*" >&2; }
fail() { printf '[dev-install] ERROR: %s\n' "$*" >&2; exit 1; }

# Must run inside the devcontainer
if [ -z "${LOCAL_DEVCONTAINER:-}" ] && [ ! -f "/.dockerenv" ] && [ ! -d "/workspaces" ]; then
  fail "Run this INSIDE the devcontainer."
fi

MODE=install  # values: install|check
for arg in "$@"; do
  case "$arg" in
    --check|--preflight) MODE=check ;;
    --install) MODE=install ;;
  esac
done

ensure_just() {
  if command -v just >/dev/null 2>&1; then
    return 0
  fi
  if [ "$MODE" = check ]; then
    warn "just missing (root Justfile tasks won't be available)."; return 1
  fi
  log "Installing just"
  mkdir -p "$HOME/.local/bin"
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --to "$HOME/.local/bin" >/dev/null 2>&1 || fail "failed to install just"
}

ensure_vscode_cli() {
  if command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q "tunnel"; then
    return 0
  fi
  [ "$MODE" = check ] && { warn "VS Code CLI (with 'tunnel') missing."; return 1; }
  log "Installing VS Code CLI"
  mkdir -p "$HOME/.local/bin"
  local arch os url tmpdir
  arch="$(uname -m)"; os="cli-linux-x64"
  if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then os="cli-linux-arm64"; fi
  url="https://update.code.visualstudio.com/latest/${os}/stable"
  tmpdir="$(mktemp -d)"
  if curl -fsSLo "${tmpdir}/cli.tar.gz" "$url" \
    && tar -xzf "${tmpdir}/cli.tar.gz" -C "$tmpdir" code; then
    install -m 0755 "${tmpdir}/code" "$HOME/.local/bin/code"
  else
    rm -rf "$tmpdir"; fail "failed to install VS Code CLI"
  fi
  rm -rf "$tmpdir"
}

ensure_cloudflared() {
  if command -v cloudflared >/dev/null 2>&1; then
    return 0
  fi
  [ "$MODE" = check ] && { warn "cloudflared missing."; return 1; }
  if ! command -v apt-get >/dev/null 2>&1; then
    fail "apt-get unavailable; cannot install cloudflared automatically."
  fi
  log "Installing cloudflared via Cloudflare APT repo"
  local key=/usr/share/keyrings/cloudflare-main.gpg
  local list=/etc/apt/sources.list.d/cloudflared.list
  sudo mkdir -p --mode=0755 "$(dirname "$key")"
  if [ ! -f "$key" ]; then
    curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee "$key" >/dev/null || fail "unable to fetch Cloudflare APT key"
  fi
  if ! sudo test -f "$list" || ! sudo grep -q "pkg.cloudflare.com/cloudflared" "$list"; then
    echo "deb [signed-by=${key}] https://pkg.cloudflare.com/cloudflared focal main" | sudo tee "$list" >/dev/null
  fi
  sudo apt-get update -y >/dev/null 2>&1 || fail "apt-get update failed"
  sudo apt-get install -y cloudflared >/dev/null 2>&1 || fail "apt-get install cloudflared failed"
}

ensure_wrangler() {
  if command -v wrangler >/dev/null 2>&1; then
    return 0
  fi
  [ "$MODE" = check ] && { warn "wrangler missing."; return 1; }
  if ! command -v npm >/dev/null 2>&1; then
    warn "npm not available; skip wrangler install."; return 0
  fi
  log "Installing Cloudflare wrangler CLI"
  npm i -g wrangler >/dev/null 2>&1 || warn "failed to install wrangler; install manually if needed"
}

preflight_checks() {
  local err=0
  if ! command -v npx >/dev/null 2>&1; then log "  - npx: MISSING (install Node.js)"; err=1; else log "  - npx: ok"; fi
  if command -v code >/dev/null 2>&1 && code --help 2>/dev/null | grep -q tunnel; then log "  - code: ok (tunnel supported)"; else log "  - code: missing or no tunnel"; err=1; fi
  if command -v cloudflared >/dev/null 2>&1; then log "  - cloudflared: ok"; else log "  - cloudflared: MISSING"; err=1; fi
  local CF_TUNNEL_L=${CF_TUNNEL:-vibekanban}
  local CF_CONF="${CLOUDFLARED_CONFIG:-$HOME/.cloudflared/config-${CF_TUNNEL_L}.yml}"
  if [ -f "$CF_CONF" ]; then log "  - cloudflared config: $CF_CONF"; else log "  - cloudflared config missing at $CF_CONF"; err=1; fi
  if cloudflared tunnel info "$CF_TUNNEL_L" >/dev/null 2>&1; then log "  - cloudflared tunnel '$CF_TUNNEL_L': found"; else log "  - cloudflared tunnel '$CF_TUNNEL_L': NOT FOUND (run owner-cloudflare-setup)"; err=1; fi
  return $err
}

case "$MODE" in
  install)
    ensure_just
    ensure_vscode_cli
    ensure_cloudflared
    ensure_wrangler
    ;;
  check)
    ensure_just || true
    log "Preflight checks:"
    if ! preflight_checks; then
      fail "Preflight failed. Fix the reported issues and re-run."
    fi
    ;;
esac

log "Done ($MODE)."
