# Project Structure Guide

**Scope.** This guide documents the current repository layout and artefact ownership. Historical structures are intentionally omitted.

**Refresh cadence.** Revisit this page after major directory reshuffles, dependency migrations, or updates to the atlas/artefact pipelines so new contributors always land on an accurate map.

## Repository map

- `/` — Workspace root for day-to-day development.
  - `AGENTS.md` — Canonical policy and workflow reference for all contributors.
  - `Justfile` — Task runner wrapping linting, typing, testing, docs, and publishing flows.
  - `mkdocs.yml` — Site configuration, navigation, and strict link validation settings.
  - `pyproject.toml` / `uv.lock` — Project metadata and pinned dependency set managed via uv.
- `src/`
  - `viterbo/` — Library package exposed to downstream experiments.
    - `capacity/`, `cycles/`, `volume/`, `spectrum.py`, `systolic.py` — Production solvers following the shared tolerance policy in `numerics.py`.
    - `atlas.py` — Helpers for the modern Hugging Face-backed atlas pipeline.
    - `_wrapped/` — Thin interoperability adapters around non-JAX primitives (e.g. SciPy, Qhull).
    - `experiments/` — Reusable experiment utilities that consume the public namespace.
    - `py.typed` — Signals downstream tools that type information ships with the package.
- `tests/`
  - `viterbo/` — Unit and integration tests structured by feature, each annotated with exactly one goal marker and suite tier.
  - `performance/viterbo/` — Benchmark harnesses aligned with the test fixtures.
  - `_baselines/` — JSON snapshots used to guard performance regressions.
- `docs/`
  - `briefs/` — Dated briefs, ADRs, and workflow guides with YAML front matter; authored per the [brief workflow](briefs/2025-10-12-workflow-brief-authoring.md).
    - `subtasks/` — Focused execution notes for active subtasks (including this guide’s parent task).
  - `project-structure.md` — This overview; MkDocs publishes the page under “Project Guide”.
  - `mkdocs.yml` references this directory tree to build the public site.
- `artefacts/`
  - `datasets/` — Curated dataset exports ready for publishing or local inspection; ensure derived data is reproducible from scripts or briefs before committing.
  - `models/` — Saved parameters or checkpoints associated with experiments; keep files lightweight and documented in briefs.
  - `published/` — Canonical artefacts mirrored to external endpoints (e.g. Hugging Face); contents should remain immutable once released.
  - Retention: commit only reproducible, curated assets. Temporary or oversized artefacts belong in scratch storage (`tmp/`) or external buckets.
- `notebooks/`
  - Script-mode notebooks (`*.py`) maintained via Jupytext-compatible cell markers for deterministic diffs.
  - `proposed/` — Staging area for exploratory notebooks awaiting promotion to the main tree.
- `scripts/` — Executable experiment and maintenance scripts; invoke via `uv run python scripts/<name>.py` or the relevant Just task.
- `mail/` — Weekly progress mail scaffolds and templates following the cadence described in `mail/README.md`.
- `thesis/` — LaTeX source for the written dissertation (`main.tex`, chapter includes, figures, bibliography).
- `typings/` — Custom type stubs required by Pyright (e.g. JAX overlays).
- `tmp/` — Ignored scratch space for local experiments; do not commit contents.

## Artefact formats and conventions

- **Briefs and workflow notes** — Markdown with YAML front matter capturing `status`, `created`, `workflow`, and `summary`. Follow the [brief authoring checklist](briefs/2025-10-12-workflow-brief-authoring.md) and store derivative notes under `docs/briefs/`.
- **Subtasks** — Dated Markdown under `docs/briefs/subtasks/` capturing scoped execution plans. They link back to their parent briefs/tasks and feed into status reviews.
- **Datasets and models** — For each asset under `artefacts/`, include a README or link back to the governing brief/ADR that explains provenance, schema, and publishing status. Prefer open, columnar formats (Parquet/JSON) and avoid raw dumps without reproducibility notes.
- **Notebooks** — Maintain `.py` notebooks with explicit section headers and cell markers. When translating into reports or briefs, capture conclusions in the relevant document rather than relying on notebook narratives.
- **Benchmarks** — Performance outputs live under `tests/performance/` and emit `.benchmarks/` artefacts during CI. Refresh baselines via dedicated tasks and record deviations in the [task evaluation workflow](briefs/2025-10-12-workflow-task-evaluation.md).

## Navigation and publishing

- MkDocs builds strictly against `mkdocs.yml`; add new guides here and link to the relevant briefs or ADRs using relative paths to keep the site portable.
- The public site (`just docs-build`) validates internal and external links via `htmlproofer`. Run this check before publishing structural updates.
- Briefs, subtasks, and artefact directories should reference each other explicitly so contributors can trace provenance without relying on chat context.

## Keeping this guide current

- Trigger a refresh when adding new top-level directories, migrating datasets, or deprecating existing pipelines.
- Cross-reference new ADRs or briefs that materially change directory ownership.
- Record any expected future migrations in the owning brief instead of speculating here; this page remains a snapshot of the present structure.
