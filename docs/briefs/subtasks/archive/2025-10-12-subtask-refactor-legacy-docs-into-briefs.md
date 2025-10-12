---
status: retired
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

## Execution notes (2025-02-14 update)

- Migrated legacy workflow/policy/reference documents into briefs:
  - Project goal and collaboration guidance now live in `docs/briefs/2025-10-12-policy-project-goal.md` and `docs/briefs/2025-10-12-workflow-codex-collaboration.md`.
  - Thesis topic, reading list, symplectic quantity reference, and algorithm catalogues moved to dedicated briefs under `docs/briefs/`.
  - The symplectic reading archive index now resides in `docs/briefs/2025-10-12-workflow-reading-archive.md`.
- Removed the corresponding legacy files from `docs/` and `docs/papers/README.md`.
- Updated `mkdocs.yml` navigation to point at the new briefs and verified no stale references remain.

## Acceptance

- Legacy workflow, policy, and reference content lives under `docs/briefs/` with proper front matter and accurate cross-links.
- Legacy files under `docs/` are removed once migrated; no stale links remain in the tree, MkDocs nav, or briefs.
- MkDocs build green with updated nav.

