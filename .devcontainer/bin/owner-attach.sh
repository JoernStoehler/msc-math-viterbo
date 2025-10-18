#!/usr/bin/env bash
set -euo pipefail

# owner-attach — Shortcut to the unified admin orchestrator.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

INTERACTIVE_FLAG=""
for a in "$@"; do
  case "$a" in
    --interactive|--tmux) INTERACTIVE_FLAG="--interactive" ;;
  esac
done

exec bash "${repo_root}/.devcontainer/bin/admin" --workspace "${WORKSPACE_DIR}" \
  attach ${INTERACTIVE_FLAG}
