---
name: experiments-notebooks-artefacts
description: Use for experiments, Jupytext-managed notebooks, and reproducible non-code artefacts.
last-updated: 2025-10-17
---

# Experiments, Notebooks, Artefacts

## Instructions
- Prefer Jupytext `.py` notebooks; preserve front-matter metadata on edits to maintain round-trip safety.
- Keep heavy outputs under `artefacts/` and link from task notes; avoid committing large binaries.
- Make runs deterministic when feasible (fixed seeds, stable inputs) and document environment details for reproducibility.
- Escalate when notebooks become system-critical; consider migrating code paths into `src/viterbo/` with tests.

## Scope

Applies when editing or creating files under `notebooks/`, including Jupytext-synced `.py` notebooks and any associated artefacts saved during exploration.

## Jupytext Workflow

1. Preserve the front-matter metadata at the top of every `.py` notebook—do not reformat or remove it. The metadata keeps pairing with optional `.ipynb` files.
2. Use Jupytext’s command-line or editor integration to sync `.ipynb` representations only when necessary for review; prefer the `.py` format for diffs.
3. If you generate a fresh notebook, copy an existing header to ensure consistent metadata (language info, kernelspec, authorship tags).

## Execution Policy

- Run notebooks inside the devcontainer to match the project environment. Avoid local host-specific dependencies.
- Clear transient outputs that are not essential to comprehension. Keep final visualizations or tabular summaries when they capture critical findings.
- Capture random seeds or generator states in the notebook to support reproducibility.

## Artefact Storage

- Large outputs (plots, CSVs, checkpoints) belong under `artefacts/` with meaningful naming; link them from notebook markdown cells rather than embedding large binaries.
- For temporary scratch data, use `tmp/` and avoid committing the files.
- Summarize enduring insights in `docs/` or relevant skills once validated so they inform future agents.

## Review & Sharing

- Note in task updates whether a notebook requires reviewer attention and highlight key cells (by number or heading).
- Avoid committing datasets unless explicitly whitelisted; reference external sources via links and describe preprocessing steps in markdown.

## Related Skills

- `repo-onboarding` — confirms early workflow steps before editing notebooks.
- `collaboration-reporting` — guides how to document findings drawn from notebooks.
- `performance-discipline` — use when profiling results from notebook explorations drive code changes.
