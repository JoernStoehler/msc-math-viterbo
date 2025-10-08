---
status: adopted
created: 2025-10-08
workflow: policy
summary: Authoring guidelines for the flat docs/briefs system and situational workflow notes.
---

# Authoring Briefs and Workflow Notes

This note replaces the legacy `docs/tasks/` template. Use it when drafting new briefs or workflow
notes so that the collection under `docs/briefs/` stays consistent and easy to navigate.

## 1. File layout and naming

- Keep every brief directly under `docs/briefs/` — no nested folders.
- Name files `YYYY-MM-DD-workflow-slug.md`. The `workflow` token should capture the document type:
  - `task`, `experiment`, or `dataset` for execution-focused briefs.
  - `workflow` or `policy` for process guidance like this note.
  - `adr` when recording adopted decisions.
- Use lowercase words in the slug separated by hyphens (e.g., `2025-10-08-task-support-function`).
- Place long-form or situational instructions (e.g., special data collection steps) into their own
  dated brief rather than reviving the `docs/tasks/` hierarchy.

## 2. Front matter conventions

Every brief starts with YAML front matter declaring:

```yaml
---
status: <draft|proposed|in-progress|blocked|adopted|retired>
created: YYYY-MM-DD
workflow: <task|experiment|dataset|workflow|policy|adr|other>
summary: One-sentence description that surfaces in search results.
---
```

Add optional keys when useful:

- `updated`: latest major revision.
- `owners`: comma-separated list of people responsible.
- `tags`: array of freeform labels (`[systolic, datasets]`).

## 3. Recommended structure

Briefs should be short, skimmable, and outcome-driven. Suggested sections:

1. **Context** — current situation, links to code, or math references required to understand the
   work.
1. **Objectives** — bullet list describing what success delivers and what artefacts will be
   produced.
1. **Execution** — concrete steps, checkpoints, and validation commands. Highlight where to record
   metrics or outputs.
1. **Dependencies / Unlocks** — upstream prerequisites plus downstream work that will rely on the
   results. Use inline bullet lists rather than Mermaid graphs.
1. **Status log** — optional dated bullet list for progress updates. Prefer short entries to keep the
   document readable; if execution notes become long, spin out a separate workflow brief.

ADRs can follow the classic structure (`Status`, `Context`, `Decision`, `Consequences`).

## 4. Referencing other documents

- Link to `AGENTS.md` when reiterating policy decisions instead of copying text.
- Point to math references under `docs/` or `docs/papers/` for definitions and proofs.
- Cross-link briefs using relative paths so navigation works both in the repo and MkDocs.

## 5. Hand-off expectations

- Each brief should make it clear what counts as "done" and how to validate the result (tests,
  benchmarks, datasets, or theoretical checks).
- Capture open questions or risks at the end. Future agents can either resolve them or fork a new
  brief with updated status.
- When a brief is superseded, set `status: retired`, add a short note pointing to the replacement,
  and remove it from the MkDocs navigation if it no longer needs prominent placement.

Following these conventions keeps the docs tree lean and ensures onboarding agents can discover the
right guidance without wading through obsolete material.
