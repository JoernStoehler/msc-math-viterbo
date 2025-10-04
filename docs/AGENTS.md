# AGENTS.md

Purpose (fact): This file governs contributions under `docs/`.

## Conventions (facts)

- Follow the repository-wide policies in `/AGENTS.md`.
- Prefer concise Markdown notes that document decisions, math references, and task briefs.
- When adding new numbered documents, reuse the existing `NN-title.md` pattern and update cross-references where they add context (e.g., roadmap entries or task briefs). A full index is optional.
- Docs should link to source modules or tests when describing behaviour; include relative paths when possible.
- Avoid committing rendered artefacts (PDFs, images) unless a task explicitly requests them.

## Tooling (facts)

- Formatting: rely on `ruff format` for Markdown files touched by code-formatting automation.
- Validation: `make ci` already runs the same checks for docs as for code (formatting, lint, waivers, typecheck, tests).

