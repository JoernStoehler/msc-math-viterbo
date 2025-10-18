---
name: operating-environment
description: This skill should be used when operating the development environment, CLI tools, and PDF ingestion workflows.
last-updated: 2025-10-18
---

# Operating the Environment

## Instructions
- Run `just checks` for a quick health pass (format, lint, type, smoke tests); use `just ci` before PRs.
- Use `uv run ...` for commands needing the Python environment; avoid manual pip installs.
- Follow the PDF ingestion steps below for `mail/private/` and prefer Markdown summaries alongside originals.
- If environment deviates from `docs/environment.md`, document differences in task notes and escalate for persistent gaps.

## Supported Stack

- Python 3.12 with PyTorch 2.x (CPU baseline; optional CUDA only in models).
- C++17 with pybind11 for performance-critical extensions.
- Development happens inside the project owner’s devcontainer; see `skills/operating-devcontainer.md` for lifecycle scripts.

## Core Commands

- See `skills/using-core-tooling.md` for the minimal command set you’ll use on most tasks (checks, lint/fix, tests, and environment glue).
- Editors: use Pyright for fast type feedback and Ruff for lint/format.
- Dependency lockfile: commit `uv.lock` when dependencies change.

## PDF Ingestion Workflow

1. Convert PDFs to Markdown using:
   ```
   pdftotext -layout -nopgbrk input.pdf output.md
   ```
2. Store the `.md` alongside the original under `mail/private/` (ignored by Git).
3. If `pdftotext` is unavailable, use a Python fallback (e.g., `pypdf`) and keep the Markdown summary.
4. Apply OCR (`tesseract`) only for scanned PDFs when necessary.

## Shell Practices

- Stream file contents in ≤250-line chunks to avoid truncation.
- Document any environment mismatches relative to `docs/environment.md` in task notes.
- Avoid manual pip installs; rely on `uv` and the devcontainer scripts for dependency management.

## Related Skills

- `operating-devcontainer` — host/container start-stop procedures.
- `always` — startup checklist and command quick reference.
- `working-with-notebooks` — reproducibility when working in notebooks.
