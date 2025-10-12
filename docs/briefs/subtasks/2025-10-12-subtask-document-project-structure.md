---
status: draft
created: 2025-10-12
workflow: task
summary: Plan the documentation overhaul describing the evolved project structure and artefacts.
---

# Subtask: Document the modern project structure

## Context

- The repository layout has evolved (e.g. HF-backed atlas, new notebooks, briefs) and the README/docs are lagging behind.
- Contributors need a clear map of key directories, artefact stores, and file formats to onboard quickly.
- Existing briefs (e.g. workflow authoring) provide partial guidance but no consolidated overview.

## Objectives (initial draft)

- Produce an up-to-date project structure guide highlighting critical paths (src, tests, docs, artefacts, notebooks).
- Document standard artefact formats (briefs, datasets, notebooks, benchmarks) and where to find them.
- Integrate the guide into the documentation site (MkDocs) for discoverability.

## Deliverables (tentative)

- New or updated doc page under `docs/` (or README section) describing the structure.
- Cross-links to relevant briefs/ADRs for deeper context.
- Optional nested-list summary (no diagrams or tables) highlighting key directories and responsibilities.

## Dependencies

- Requires up-to-date knowledge of repository layout, including recent additions under `docs/briefs/`, `artefacts/`, and atlas datasets.
- Should align with existing MkDocs navigation (`mkdocs.yml`) to ensure the new page is discoverable.
- May depend on other documentation efforts (e.g., workflow briefs) to avoid duplication; coordinate references accordingly.

## Acceptance criteria (to validate completion)

- A MkDocs page under `docs/` introduces the project structure, highlights key directories, and explains artefact retention practices without referencing external chat context.
- Navigation updates in `mkdocs.yml` surface the new page alongside related guides.
- Links to briefs or ADRs are verified and use relative paths.
- The document explicitly states scope (current structure only) and employs nested lists for directory breakdowns.

## Decisions and constraints

- Publish the guide within `docs/` (MkDocs), not under `docs/briefs/` or the root README.
- Aim for medium-grain detail: enough specifics to onboard new contributors without enumerating every file; author judgement applies.
- Focus entirely on the current structure. Explicitly note near the top that historical layout is intentionally omitted.
- Capture what belongs inside `artefacts/` (and what does not), including any retention/cleanup expectations.
- Prefer nested lists for structure breakdowns; avoid diagrams and tables to keep maintenance light.
- Exclude workflow instructions for adding briefs/subtasks from this guide; plan a separate workflow brief to cover that process.

## Open Questions

1. None at this stage; confirm once atlas documentation and workflow briefs stabilise.

## Notes

- Review and update the guide after major structural refactors; note the expected refresh cadence in the document.
