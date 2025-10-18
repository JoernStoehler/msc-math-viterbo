#!/usr/bin/env bash
set -euo pipefail

# owner-up — Shortcut to the unified admin orchestrator.

# Resolve workspace folder (defaults to repo root)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

# Extract flags we want to apply to `start` (e.g., --interactive)
ARG_AFTER_START=()
for a in "$@"; do
  case "$a" in
    --interactive) ARG_AFTER_START+=("--interactive") ;;
  esac
done

exec bash "${repo_root}/.devcontainer/bin/admin" --workspace "${WORKSPACE_DIR}" \
  up preflight start ${ARG_AFTER_START[*]:-} status
