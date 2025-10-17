---
name: skill-authoring
description: Create or update skills that align with Anthropic’s spec, mix facts with procedures, and stay concise.
last-updated: 2025-10-17
---

# Skill Authoring Guide

Follow these steps when creating or updating any `skills/*.md` file so your document stays compatible with Anthropic’s `SKILL.md` packaging and easy for Claude/Codex agents to use.

## Instructions

1) Metadata and naming
- Create `skills/<slug>.md` using lowercase, hyphenated names that match the `name` field.
- Add YAML frontmatter:
  ```markdown
  ---
  name: <slug>
  description: Use this when …
  last-updated: YYYY-MM-DD
  ---
  ```
- Keep `description` to one sentence explaining what the skill covers and when Claude should load it.
- Optional keys (`license`, `metadata`, `allowed-tools`) must match Anthropic’s spec.
- Design for three-stage loading (per Anthropic): metadata selects; the body loads in full; references load on demand.

2) Write the body
- Begin with instructions immediately. Do not include triage like “read this if …”—selection already happened via `description`.
- Mix facts with actions: state key facts (layout, invariants, policies) next to the steps that use them.
- Use imperative voice for procedures (“Run…”, “Escalate…”). Use declarative sentences for facts when clearer.
- Highlight guardrails and escalation triggers in their own bullets.
- Keep the main file concise; move big examples or templates to references and link them.
- End with a short “Related Skills” section to aid navigation.

3) Validate
- Run `just lint` (or `uv run python scripts/load_skills_metadata.py --quiet`) to confirm valid frontmatter.
- Check internal and external links.
- Bump `last-updated` whenever instructions change behavior.
- Note major additions in task notes or PR summaries.

## Checklist
- [ ] Filename matches `name`, lowercase hyphenated.
- [ ] `description` states scope and trigger in one sentence.
- [ ] Body starts with instructions and mixes facts with actions.
- [ ] Guardrails and escalation triggers called out.
- [ ] Related Skills section present (when applicable).
- [ ] Lint/metadata validation passes.
- [ ] `last-updated` bumped.

## References
- Anthropic Skills Spec (agent_skills_spec.md): https://github.com/anthropics/skills/blob/main/agent_skills_spec.md
- Creating custom skills (Help Center): https://support.claude.com/en/articles/12512198-creating-custom-skills
- `skill-creator` example repo: https://github.com/anthropics/skills/tree/main/skill-creator
