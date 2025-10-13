---
status: draft
created: 2025-10-12
workflow: task
summary: Outline the work to finish the four Monday notebooks that visualise the atlas dataset.
---

# Subtask: Complete Monday visualisation notebooks

## Context

- Four placeholder notebooks live under `notebooks/proposed/`: `dataset_pipeline.py`, `atlas_umap.py`, `polytope_tables.py`, and `minimal_action_cycle.py`.
- Each notebook assumes an atlas dataset is available. Today only the `atlas_tiny` preset exists; the envisaged `atlas_small` snapshot still needs to be built once the expanded builder lands.
- The notebooks should demonstrate realistic downstream consumption of the atlas artefacts (tables, embeddings, trajectory visualisations).

## Objectives (initial draft)

- Implement the full data loading, processing, and plotting flow in each notebook, targeting non-interactive execution under Codex CLI with MCP support.
- Ensure notebooks share a consistent configuration layer (paths, dataset version) to simplify reruns.
- Produce exportable artefacts (e.g. static images, summary tables) that can be referenced in reports.

## Deliverables (tentative)

- Updated notebooks with runnable code cells and minimal Markdown framing.
- Snapshot of generated figures/tables saved under `artefacts/notebooks/monday/` (or other agreed location).
- Optional helper utilities in `src/viterbo` if needed for reuse.

## Dependencies

- Depends on availability of an `atlas_tiny` dataset (existing) and future `atlas_small` export once the dataset build task ships.
- Shares configuration patterns with any future documentation describing dataset consumption.
- Requires Codex CLI MCP integration if authors intend to iterate interactively, though notebooks must remain runnable without it.

## Acceptance criteria (to validate completion)

- Each notebook executes top-to-bottom in a fresh kernel using `jupytext` without manual edits, saving artefacts to the agreed folder.
- Common configuration (paths, dataset selectors) is factored to avoid duplication and documented in a short README or notebook preamble.
- Generated figures/tables are stored under version-controlled artefact directories with clear filenames and captions in the notebooks.
- Headless execution produces no interactive prompts and completes within reasonable time on the devcontainer hardware.

## Decisions and constraints

- Stay on Jupytext `.py` notebooks for now; they can be executed top-to-bottom without a persistent kernel and still support presentation in VS Code later.
- Default dataset: `atlas_tiny` until larger presets (e.g., `atlas_small`) become available.
- Plotting stack: Matplotlib is sufficient; extend only if specific needs arise.
- Rendering must remain headless and avoid interactive viewers (saving figures to disk instead).
- The notebook content itself serves as the deliverable—no separate HTML/Markdown reports required.
- Runtime limits follow typical devcontainer expectations; keep notebooks efficient but no hard cap beyond common sense.

## Open Questions

1. None currently; revisit once atlas dataset interfaces settle or if additional presentation requirements emerge.

## Notes

- Maintain notebook headers with a brief summary, runtime expectations, and dataset version pinning for future maintainers.
