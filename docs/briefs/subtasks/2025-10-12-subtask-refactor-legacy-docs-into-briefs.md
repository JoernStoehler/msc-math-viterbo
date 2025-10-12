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

- Identify non-reference documents under `docs/` (workflow/process/ADR-like) and convert them into `docs/briefs/` entries.
- Keep reference pages (mathematical references, reading lists) in place.
- Update MkDocs navigation accordingly.

## Plan

1. Audit `docs/` and classify files as reference vs workflow/policy.
2. For workflow/policy, create briefs with front matter and concise summaries; cross-link from the original path until consumers switch.
3. Update `mkdocs.yml` to point to briefs; optionally keep legacy paths for a deprecation period.
4. Remove or retire legacy pages once the nav and links are stable.

## Acceptance

- Workflow/policy content lives under `docs/briefs/` with proper front matter.
- Reference content remains in `docs/`.
- MkDocs build green with updated nav.

