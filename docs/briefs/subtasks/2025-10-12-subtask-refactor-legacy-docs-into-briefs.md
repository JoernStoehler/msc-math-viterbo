---
status: proposed
created: 2025-10-12
workflow: task
summary: Refactor legacy docs/* workflow/policy notes into modern briefs.
---

# Subtask: Refactor legacy docs into briefs

## Context

- Some older workflow/policy notes live under `docs/` outside `docs/briefs/`.
- We want a consistent modern briefs structure for tasks/ADRs/workflows.

## Objectives

- Identify every legacy workflow, policy, and reference document under `docs/` that should live in the briefs tree.
- Convert each document into one or more `docs/briefs/` entries with the appropriate front matter, metadata, and cross-links.
- Update MkDocs navigation and in-page links accordingly.

## Plan

1. Audit `docs/` and inventory legacy workflow, policy, and reference notes that pre-date the briefs structure.
2. For each item, determine whether it maps to an existing brief or needs a new one; migrate the content with updated front matter, concise summaries, and explicit references to related briefs.
3. Remove the legacy files once their content is represented in `docs/briefs/`, and update any relative links or embeds that pointed to the legacy path.
4. Update `mkdocs.yml` to point to the new briefs; remove obsolete nav entries.
5. Confirm no orphaned references remain and clean up any redirects or placeholders.

## Status check (2025-02-14 audit)

- `docs/01-project-goal.md`, `docs/05-working-with-codex.md`, `docs/11-math-thesis-topics.md`, `docs/12-math-reading-list.md`, `docs/13-symplectic-quantities.md`, `docs/convex-polytope-cehz-capacities.md`, and `docs/convex-polytope-volumes.md` still sit at the top level of `docs/` and retain the pre-brief format.
- Topic folders such as `docs/papers/**` continue to host LaTeX/Markdown notes about individual references and have not been migrated into briefs.
- MkDocs navigation still references the legacy paths above.
- Conclusion: the migration remains outstanding; the acceptance criteria below are not yet met.

## Acceptance

- Legacy workflow, policy, and reference content lives under `docs/briefs/` with proper front matter and accurate cross-links.
- Legacy files under `docs/` are removed once migrated; no stale links remain in the tree, MkDocs nav, or briefs.
- MkDocs build green with updated nav.

