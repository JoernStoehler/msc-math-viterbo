---
status: ready
created: 2025-10-21
owner: Project Owner Jörn
advisors: Kai Cieliebak
summary: Run-of-show for today’s meetup with links to generated artefacts and demo commands.
---

# Meetup Run-of-Show — 2025-10-21

This agenda focuses on fast, visual demos with reproducible outputs already generated. All items run CPU-only with float64 discipline.

## Quick Checks (2 min)
- Code health: `just checks` green (lint, type, smoke tests).
- Docs: `just docs-build` green (MkDocs strict).

## Demos (10–12 min total)
- AtlasTiny v1 — dataset + figures (4–5 min)
  - Artefacts: `artefacts/datasets/atlas-tiny/v1/` (Parquet + metadata + README)
  - Quick plots: `artefacts/published/atlas_tiny_explorer/` (4 PNGs)
  - Rebuild: `uv run python notebooks/atlas_tiny_build.py`
  - Explore: `uv run python notebooks/atlas_tiny_explorer.py`

- 4D counterexample (pentagon×rot90) — Q/P projection (3–4 min)
  - Show: `artefacts/published/atlas_tiny_explorer/pentagon_product_counterexample_4d_cycle_qp.png`
  - Talking points: minimal_action_cycle, backend provenance, time_* fields

- Invariance sanity (1–2 min)
  - 2D cycles and areas stable under translation/scale; tensors remain float64 CPU.

## Decisions / Asks (3–4 min)
- Confirm scope for next week’s oriented-edge certified runs (timeout budgets).
- Decide whether to promote AtlasTiny as public demo dataset (Parquet schema freeze).
- Approve adding links to meeting artefacts in docs nav.

## Appendix — Commands
- Build dataset: `uv run python notebooks/atlas_tiny_build.py`
- Explore figures: `uv run python notebooks/atlas_tiny_explorer.py`
- Full checks: `just ci` (longer)
- Clean artefacts: `just clean-artefacts`

## References
- Dataset README: `artefacts/datasets/atlas-tiny/v1/README.md`
- Explorer outputs: `artefacts/published/atlas_tiny_explorer/`
- Prep task (full checklist): `docs/meeting/2025-10-21-prep-task.md`

