---
name: environment-tooling
description: Manage the devcontainer, command-line tooling, and utility workflows (including PDF ingestion).
last-updated: 2025-10-17
---

# Environment & Tooling

## Supported Stack

- Python 3.12 with PyTorch 2.x (CPU baseline; optional CUDA only in models).
- C++17 with pybind11 for performance-critical extensions.
- Development happens inside the project owner’s devcontainer; see `skills/devcontainer-ops.md` for lifecycle scripts.

## Core Commands

- `just checks` — run lint, type checks, and smoke tests.
- `just lint` — Ruff lint + skill metadata validation (`scripts/load_skills_metadata.py --quiet`).
- `just type` — Pyright (basic) against `src/viterbo`.
- `just test` — incremental pytest smoke tier.
- `just ci` — CI parity (lint, type, tests) before handoff or PR creation.
- `uv run ...` — execute Python scripts/tests with dependencies resolved via `uv.lock`.
- `rg` — preferred code/search tool; stream ≤250 lines when reading in shell.

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

- `devcontainer-ops` — host/container start-stop procedures.
- `repo-onboarding` — startup checklist and command quick reference.
- `notebook-etiquette` — reproducibility when working in notebooks.
