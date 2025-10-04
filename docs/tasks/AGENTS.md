# AGENTS — docs/tasks/

## Scope
This file governs every document under `docs/tasks/`, including lifecycle
subdirectories (`draft/`, `rfc/`, `scheduled/`, `completed/`) and overview files.
Follow these rules when authoring, editing, or moving task briefs.

## 1. Naming and location
- Place new briefs inside one of the lifecycle subdirectories using the pattern
  `YYYY-MM-DD-slug.md`. The date reflects when the brief entered the queue.
- Overview documents that apply to the whole portfolio (e.g., methodology,
dependency tables) live directly in `docs/tasks/` with numeric prefixes to convey
reading order.
- Do **not** use symlinks for briefs; link to the canonical file instead.

## 2. Lifecycle workflow
1. **Draft**: initial sketch with open questions. Update `Status: Draft` and the
   `Last updated` timestamp.
2. **RFC**: ready for maintainer review; capture requested feedback inline.
3. **Scheduled**: accepted and awaiting execution. Keep dependencies, risks, and
   checkpoints current.
4. **In Progress**: optionally noted during execution; update status and
   timestamps in place.
5. **Completed**: move the file to `completed/` and append a short outcome
   summary (success, partial, blocked, etc.).

Record transitions in the "Follow-on work" section rather than editing history.

## 3. Template usage
- Use `docs/tasks/template.md` as the starting point for every new brief.
- Preserve all section headings unless a section is genuinely irrelevant; delete
  empty bullet lists once populated.
- Keep scope/objectives concise so executors retain flexibility for
  implementation details.

## 4. Cross-references
- Reference related material with relative links (e.g., `../02-task-portfolio.md`
  or other briefs) so the files stay portable.
- When a brief depends on roadmap items or code modules, link directly to those
  documents for fast onboarding.
- Update `docs/tasks/02-task-portfolio.md` whenever priorities or dependencies
  change.

## 5. Benchmark & testing notes
- When a brief prescribes benchmarks, specify which tier (inner/CI/deep) belongs
  where. For long-haul runs, point readers to `.benchmarks/` artefacts and note
  summarised results in the brief or a linked progress report.
- Avoid promising CI coverage that exceeds the maintainer's <5 minute target; use
  pytest markers to separate smoke vs. deep suites.

## 6. Style expectations
- Write in complete sentences with imperative clarity. Use bullet lists for
  deliverables, risks, and follow-on work.
- Keep line length ≤100 characters (Ruff format). Tables are welcome for cost or
  utility breakdowns.
- Include escalation triggers explicitly (tie them back to the root `AGENTS.md`
  policy).

Adhering to these conventions keeps the task queue maintainable and predictable
for future Codex agents.
