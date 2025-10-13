---
status: draft
created: 2025-10-12
workflow: task
summary: Inventory of exported APIs and proposal for simplifying the Viterbo surface.
---

# API simplification opportunity

## Context

The namespace cleanup moved the public surface into two focused packages: `viterbo.math` (pure
algorithms) and `viterbo.datasets2` (dataset builders and quantity calculators). The legacy modules
referenced in earlier drafts (`viterbo.atlas`, `viterbo.types`, `viterbo.polytopes`, …) no longer
exist. This brief captures the *current* inventory so follow-up simplification work can target the
actual code.

### [important] Design directives captured from maintainer feedback

1. Keep math code JAX-first and side-effect free. Expose explicit functions grouped by algorithmic
   concern (`capacity`, `spectrum`, `volume`, `geometry`, …).
2. Treat datasets as thin adapters on top of math code. Avoid re-export indirection; prefer calling
   the concrete module (`viterbo.math.capacity.bla`).
3. Document the public surface in situ (docstrings, module docs) and keep the briefs aligned with the
   real files.

## Inventory (October 2025)

### `viterbo.math`

- `geometry.py` — Half-space/vertex conversions, incidence matrices, random generators.
- `generators.py` — Canonical catalogue and sampling helpers for study polytopes.
- `capacity/` — Facet optimisation, Reeb cycles, Minkowski billiards, support-function relaxations.
- `spectrum.py` — Symplectic spectrum utilities and caching helpers.
- `symplectic.py` — Symplectic linear algebra primitives.
- `volume.py` — Reference and fast volume estimators.
- `numerics.py` — Shared tolerances and solver configuration.
- `systolic.py` — Systolic capacity utilities.

### `viterbo.datasets2`

- `atlas_tiny.py` — Streams a small HF Dataset (`atlas_tiny`) with geometry, capacity, spectrum,
  tags, and provenance columns sourced from `quantities.py`.
- `quantities.py` — Quantity calculators powering atlas rows (volume, capacity, spectrum, half-space
  conversions).

### Supporting namespaces

- `viterbo._wrapped/` — SciPy/Qhull interop shims that back selected math functions.
- `viterbo.experiments/`, `viterbo.visualization/`, `viterbo.models/` — Staging areas for promoted
  helpers (sparse at the moment).
- Tests live under `tests/viterbo/` and mirror the math/datasets split.

## Simplification themes

1. **Documented surface.** Add lightweight module docs summarising the stable entry points and link
   back to this brief for context.
2. **Import hygiene.** Ensure notebooks, scripts, and docs import from the modern modules. Flag any
   lingering references to removed namespaces for cleanup.
3. **Builder coverage.** Plan follow-on presets (`atlas_small`, …) and keep the row schema aligned
   across datasets.

## Next steps

- Draft RFCs/PRDs covering CLI/Notebook integration and future dataset presets once the code-side
  implementation begins.
- Keep this inventory updated after each simplification pass so future contributors do not need to
  reverse-engineer the namespace.
