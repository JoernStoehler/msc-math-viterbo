---
name: authoring-skills
description: This skill should be used when authoring or updating skills aligned with Anthropic’s guidance, combining facts with imperative steps.
last-updated: 2025-10-18
---

# Authoring Skills

Write skills that Claude/Codex can discover quickly, read efficiently, and execute reliably. This guide adapts Anthropic’s official patterns to our repository conventions (flat `skills/` files, loader script, and local references).

## Instructions

1) Metadata and naming
- Create `skills/<slug>.md` using lowercase, hyphenated names that match the `name` field.
- Use a gerund/action name that captures what the skill helps you do (e.g., `testing-and-troubleshooting`, `operating-environment`) rather than static nouns (e.g., `testing-workflow`, `environment-tooling`).
- Add YAML frontmatter:
  ```markdown
  ---
  name: <slug>
  description: This skill should be used when …
  last-updated: YYYY-MM-DD
  ---
  ```
- Keep `description` to one sentence that states scope and trigger in third person (“This skill should be used when…”).
- Optional keys (`license`, `metadata`, `allowed-tools`) must match Anthropic’s spec (allowed-tools is Claude Code only).
- Design for three‑stage loading (Anthropic):
  - Metadata (name + description): always visible and used for selection.
  - Body: loaded fully when the skill is chosen; avoid in‑body triage text.
  - References/resources: loaded on demand; keep the main file concise.

2) Write the body
- Begin with instructions immediately; the description already handled triage.
- Mix facts with actions: state key facts (layout, invariants, policies) next to the steps that use them.
- Use imperative voice for procedures (“Run…”, “Escalate…”). Use declarative sentences for facts when clearer.
- Highlight guardrails and escalation triggers in their own bullets.
- Keep the main file concise; move big examples or templates to references and link them.
- End with a short “Related Skills” section to aid navigation.

3) Optional resources (adapted from Anthropic)
- Prefer referencing production repo files (e.g., `scripts/`, `src/`, `tests/`, `docs/`, `.devcontainer/`) over duplicating content.
- If you truly need bundled files under skills, use:
  - `skills/resources/` for simple reference snippets or tiny assets.
  - Avoid deep nesting and duplication. For large references, include grep/search tips in the skill body.
  - Do not assume tools are installed; declare prerequisites explicitly.

4) Validate
- Run `just lint` (or `uv run python scripts/load_skills_metadata.py --check`) to confirm valid frontmatter and AGENTS.md sections are current.
- Check internal and external links.
- Bump `last-updated` whenever instructions change behavior.
- Note major additions in task notes or PR summaries.

5) Always-on skills (optional)
- To include a skill in the auto-generated “Always-On Skills” section of `AGENTS.md`, add one of:
  - `relevance: always` (preferred)
  - `always: true` or `always-on: true`

## Checklist
- [ ] Filename matches `name`, lowercase hyphenated.
- [ ] `description` states scope and trigger in one sentence.
- [ ] Body starts with instructions and mixes facts with actions.
- [ ] Guardrails and escalation triggers called out.
- [ ] Related Skills section present (when applicable).
- [ ] Lint/metadata validation passes.
- [ ] `last-updated` bumped.

## Minimal skeleton
```markdown
---
name: example-skill
description: This skill should be used when …
last-updated: 2025-10-17
---

# Example Skill

## Instructions
- Step 1: …
- Step 2: …

## Key facts
- Fact A: …
- Fact B: …

## Related Skills
- `skills/another-skill.md`
```

## Best practices
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
- Skill authoring best practices: https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices
- `skill-creator` example repo: https://github.com/anthropics/skills/tree/main/skill-creator
