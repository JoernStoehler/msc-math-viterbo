---
name: environment-tooling
description: Troubleshooting environment/tooling deviations from the golden path and keeping flows healthy.
last-updated: 2025-10-18
---

# Environment & Tooling (Troubleshooting)

## Instructions
- If the environment deviates from `docs/environment.md`, document differences in task notes and escalate persistent gaps.
- Prefer `uv run ...` for Python entry points; avoid manual pip installs.
- Use this skill to diagnose failing `just`/`uv` commands or editor tooling mismatches.

## Supported Stack

- Python 3.12 with PyTorch 2.x (CPU baseline; optional CUDA only in models).
- C++17 with pybind11 for performance-critical extensions.
- Development happens inside the project owner’s devcontainer; see `skills/devcontainer-ops.md` for lifecycle scripts.

## Diagnostics

- Verify `uv` version and environment activation; re-run `uv run python -V` to confirm.
- Inspect `Justfile` targets when a command fails; run sub-steps manually to isolate issues.
- Check editor tooling (Pyright/Ruff) aligns with project config; prefer in-repo settings.

## Core Commands

- `skills/basic-environment.md` lists the golden command palette (`just checks`, `just ci`, `just lint`, `just fix`, `just test`).
- Editors: use Pyright for fast type feedback and Ruff for lint/format; prefer the project’s in-repo settings.
- Dependency lockfile: commit `uv.lock` whenever dependencies change so other agents pick up the update.
- Provisioning already runs `just setup`; you rarely need to trigger it manually unless onboarding scripts change.

## When to Rebuild

- After dependency or base image updates fail locally, follow `devcontainer-ops` rebuild guidance.
- Avoid ad-hoc cache wipes; capture exact error and command history first.

## Shell Practices

- Stream file contents in ≤250-line chunks to avoid truncation.
- Document any environment mismatches relative to `docs/environment.md` in task notes.
- Avoid manual pip installs; rely on `uv` and the devcontainer scripts for dependency management.

## PDF Ingestion Workflow

1. Convert PDFs to Markdown using:
   ```
   pdftotext -layout -nopgbrk input.pdf output.md
   ```
2. Store the `.md` alongside the original under `mail/private/` (git-ignored).
3. If `pdftotext` is unavailable, fall back to a Python tool (`pypdf`, etc.) and keep the Markdown summary.
4. Apply OCR (`tesseract`) only for scanned PDFs; capture the command you used in task notes.

## Related Skills

- `devcontainer-ops` — host/container start-stop procedures.
- `basic-environment` — golden commands and navigation.
- `repo-layout` — structure and sources of truth.
- `experiments-notebooks-artefacts` — notebook execution environment and artefacts.
