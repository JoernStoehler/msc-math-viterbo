---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Updated authoring guidelines for briefs in the modern docs tree.
---

# Authoring Briefs (Modern)

## File layout and naming

- Keep every brief directly under `docs/briefs/` — no nested folders.
- Name `YYYY-MM-DD-workflow-slug.md`; workflows include `task`, `adr`, `workflow`, `policy`.

## Front matter

```yaml
---
status: <draft|proposed|in-progress|blocked|adopted|retired>
created: YYYY-MM-DD
workflow: <task|adr|workflow|policy>
summary: One sentence summary.
---
```

## Structure

Keep documents short and outcome-driven. Suggested sections: Context, Objectives, Execution,
Dependencies/Unlocks, Acceptance, and an optional Status Log.

## Hand-off expectations

- State what counts as “done” and how to validate (tests/benchmarks/datasets).
- When superseded, set `status: retired` and point to the replacement.

