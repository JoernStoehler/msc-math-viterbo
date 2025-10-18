#!/usr/bin/env bash
set -euo pipefail

# owner-down — Shortcut to the unified admin orchestrator.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
WORKSPACE_DIR=${WORKSPACE_DIR:-${repo_root}}

exec bash "${repo_root}/.devcontainer/bin/admin" --workspace "${WORKSPACE_DIR}" \
  stop down
