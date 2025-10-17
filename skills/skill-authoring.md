---
name: skill-authoring
description: Create or update skills that align with Anthropic’s spec, mix facts with procedures, and stay concise.
last-updated: 2025-10-17
---

# Skill Authoring Guide

## Scope

Use this skill whenever you add or revise files under `skills/`. It distills Anthropic’s published guidance so new instructions stay compatible with `SKILL.md` packaging.

## Before You Write

1. Open the canonical references:  
   - Anthropic Skills spec (`agent_skills_spec.md`) — naming, metadata keys.  
   - “Creating custom skills” help article — progressive disclosure patterns.  
   - `skill-creator` example in `anthropics/skills` — full template.  
   - `docs/creating_skills.md` in this repo — local conventions.  
2. Decide whether the content is worth a dedicated skill: group tiny tips; split only when audiences do not overlap.

## Structure & Metadata

1. Create `skills/<slug>.md` using lowercase hyphenated names.  
2. Add YAML frontmatter:
   ```markdown
   ---
   name: <slug>
   description: Use this when …
   last-updated: YYYY-MM-DD
   ---
   ```
   - Keep `description` to one sentence describing both scope and trigger.  
   - Optional keys (`license`, `metadata`, `allowed-tools`) must match the Anthropic spec.
3. Write a short “Scope” or “Purpose” section that states what problem the skill solves and when Claude should load it.

## Writing Principles

1. **Mix facts with actions.** State essential knowledge (layout, invariants, policies) and follow immediately with the steps that enforce it.  
2. **Imperative voice first.** Lead instructions with verbs (“Run…”, “Escalate…”) so agents can execute them verbatim. Use declarative sentences only when facts become awkward in imperative form.  
3. **Highlight guardrails.** Call out escalation triggers, destructive-command bans, or policy constraints in their own bullets.  
4. **Keep it tight.** Aim for a few hundred words. Offload long examples or templates into linked references.  
5. **Link related skills.** Close the file with a “Related Skills” section to aid navigation.

## Validation

1. Run `just lint` (or `uv run python scripts/load_skills_metadata.py --quiet`) to verify frontmatter.  
2. Confirm links resolve: internal (`skills/...`) and external (Anthropic docs, repo references).  
3. Bump `last-updated` whenever the instructions change behavior.  
4. Record major additions in task notes or PR summaries for visibility.

## Related Skills

- `docs/creating_skills.md` — extended rationale, references, and checklist.
- `repo-onboarding` — explains when to load skills during task startup.
