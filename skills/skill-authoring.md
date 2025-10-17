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

## Best Practices
- Concise is key: every paragraph must justify its token cost. Prefer a few hundred words; link out for deep dives.
- Progressive disclosure: keep the body focused; link references instead of embedding long content. Avoid deeply nested reference chains.
- Degrees of freedom: allow limited flexibility when multiple approaches are valid; be explicit about the preferred default path.
- Consistent terminology: reuse canonical names and shapes; avoid synonyms that might confuse pattern matching.
- Avoid time-sensitive info: prefer timeless guidance or date-stamp facts; update `last-updated` with meaningful changes.
- Avoid offering too many options: present a recommended path first; list alternates only when truly needed.
- Avoid assuming tools are installed: declare prerequisites or use `allowed-tools` (Claude Code only) when necessary.
- Path style consistency: use POSIX-style paths in examples; avoid Windows-only path forms unless the skill is Windows-specific.
- Structure long references with a simple table of contents; keep headings descriptive.
- Token budgets: keep metadata tight; keep examples small and representative.

## References
- Anthropic Skills Spec (agent_skills_spec.md): https://github.com/anthropics/skills/blob/main/agent_skills_spec.md
- `skill-creator` example repo: https://github.com/anthropics/skills/tree/main/skill-creator
 - Skill authoring best practices: https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices
