#!/usr/bin/env bash
set -euo pipefail

# owner-rebuild — Shortcut to the unified admin orchestrator.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

# Extract flags for placement
NO_CACHE_FLAG=""
INTERACTIVE_FLAG=""
for a in "$@"; do
  case "$a" in
    --no-cache) NO_CACHE_FLAG="--no-cache" ;;
    --interactive) INTERACTIVE_FLAG="--interactive" ;;
  esac
done

exec bash "${repo_root}/.devcontainer/bin/admin" --workspace "${WORKSPACE_DIR}" \
  rebuild ${NO_CACHE_FLAG} preflight start ${INTERACTIVE_FLAG} status
