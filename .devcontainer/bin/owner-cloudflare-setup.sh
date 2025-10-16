#!/usr/bin/env bash
set -euo pipefail

CF_TUNNEL=${CF_TUNNEL:-vibekanban}
CF_HOSTNAME=${CF_HOSTNAME:-vibekanban.joernstoehler.com}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
CF_CONFIG_DEFAULT="$HOME/.cloudflared/config-${CF_TUNNEL}.yml"
CF_CONFIG_PATH=${CLOUDFLARED_CONFIG:-$CF_CONFIG_DEFAULT}

log() {
  printf '[owner-cloudflare-setup] %s\n' "$*"
}

fail() {
  printf '[owner-cloudflare-setup] ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf '[owner-cloudflare-setup] WARNING: %s\n' "$*" >&2
}

command -v cloudflared >/dev/null 2>&1 || fail "cloudflared not installed."

if [ ! -s "$HOME/.cloudflared/cert.pem" ]; then
  fail "cloudflared not logged in. Run 'cloudflared tunnel login' first."
fi

if ! cloudflared tunnel info "$CF_TUNNEL" >/dev/null 2>&1; then
  fail "tunnel '${CF_TUNNEL}' not found. Create it before running this command."
fi

CF_CREDENTIALS=""
if [ -s "$HOME/.cloudflared/${CF_TUNNEL}.json" ]; then
  CF_CREDENTIALS="$HOME/.cloudflared/${CF_TUNNEL}.json"
else
  for candidate in "$HOME"/.cloudflared/*.json; do
    [ -f "$candidate" ] || continue
    CF_CREDENTIALS="$candidate"
    break
  done
fi

[ -n "$CF_CREDENTIALS" ] || fail "no tunnel credential (*.json) found in $HOME/.cloudflared."

if [ ! -f "$CF_CONFIG_PATH" ]; then
  cat >"$CF_CONFIG_PATH" <<EOF
tunnel: ${CF_TUNNEL}
credentials-file: ${CF_CREDENTIALS}
ingress:
  - hostname: ${CF_HOSTNAME}
    service: http://127.0.0.1:${FRONTEND_PORT}
  - service: http_status:404
EOF
  log "wrote config to $CF_CONFIG_PATH"
else
  log "config already present at $CF_CONFIG_PATH"
fi

if [ -n "$CF_HOSTNAME" ]; then
  if cloudflared tunnel route dns "$CF_TUNNEL" "$CF_HOSTNAME" >/dev/null 2>&1; then
    log "DNS route ensured for ${CF_HOSTNAME}"
  else
    warn "unable to ensure DNS route for ${CF_HOSTNAME}; run manually."
  fi
else
  warn "CF_HOSTNAME not set; skipping DNS route."
fi
