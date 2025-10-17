---
title: Creating Skills
---

# Creating Skills

Use the `skills/` directory to capture agent-facing workflows and the conceptual knowledge that supports them. The goal is to give another Claude/Codex agent the minimum information they need to work safely and effectively: clear metadata, task triggers, factual context, and imperative steps. Every file should mirror Anthropic's published guidance so it can be packaged as a `SKILL.md` later without surprises.

## Canonical References

- [Anthropic Skills Spec (`agent_skills_spec.md`)](https://github.com/anthropics/skills/blob/main/agent_skills_spec.md)
- [Creating custom skills (Claude help center)](https://support.claude.com/en/articles/12512198-creating-custom-skills)
- [Skills repository examples (`skill-creator`)](https://github.com/anthropics/skills/tree/main/skill-creator)
- [Agent Skills launch blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Prompt/Context engineering best practices](https://support.claude.com/en/collections/7423347-prompting-claude) — reinforces progressive disclosure and concise instructions

Refer back to these documents whenever in doubt; our local conventions extend rather than replace them.

## When to Add or Update a Skill

- Consolidate correlated workflows into one file (e.g., devcontainer lifecycle + CLI cheats) rather than scattering instructions.
- Split only when two audiences rarely need both workflows in the same task.
- Update the `last-updated` field whenever a meaningfully new policy lands; note the change in the file body.
- If knowledge is transient or tiny (one-line tips), group it under a shared skill instead of creating noise.

## File Layout and Metadata

- Location: `skills/<slug>.md`
- Filename doubles as the skill name; keep it short, hyphenated, and lowercase.
- Frontmatter template:
  ```markdown
  ---
  name: <repeat-or-human-friendly-name>
  description: One-sentence trigger for when to load the skill
  last-updated: YYYY-MM-DD
  ---
  ```
- Required metadata keys: `name`, `description`. Optional keys (`last-updated`, `license`, `metadata`, `allowed-tools`) must follow the Anthropic spec.
- Write the description as a single sentence explaining *what* the skill covers and *when* Claude should load it (“Use this when…”).
- Body: combine factual context (e.g., directory layout, invariants) with imperative steps. Push lengthy references or assets into dedicated files and link them.

## First-Action Script

- After reading `AGENTS.md`, run `uv run python scripts/load_skills_metadata.py` to stream metadata into the model context.
- The script prints one summarized line per skill; no instructions are emitted, keeping the context budget small.
- If a skill is critical for every task, link it from `AGENTS.md` so agents load it explicitly after running the script.
- Automation workflows (e.g., `just lint`) pass `--quiet` to suppress summaries while still validating frontmatter.
- `just lint` fails fast if frontmatter is malformed; fix issues in metadata before focusing on content edits.

## Writing Principles

1. **State scope up front.** Begin with a “Purpose” or “Scope” section answering what the skill solves and when to use it.
2. **Mix facts and actions.** Present key knowledge (“`math/` modules must stay pure”) next to the procedure that enforces it (“Return tensors without device moves”).
3. **Use imperative voice.** Start steps with verbs so the agent can follow them verbatim. Declarative sentences are fine for facts when they are clearer than a forced imperative.
4. **Highlight guardrails.** Call out escalation triggers, destructive-command bans, or policy constraints.
5. **Link related skills.** Close with a short “Related Skills” section to help agents chain the right guides without guessing.
6. **Stay concise.** Keep the main file within a few hundred words. Offload examples, templates, or deep dives into `references/` or existing docs.

## Review Checklist

- [ ] Metadata describes when the skill applies (not just what it contains).
- [ ] Instructions use imperative voice and assume the reader is another Claude agent.
- [ ] Key facts appear alongside the steps that rely on them.
- [ ] Links point to canonical docs/scripts; avoid duplicating long-form explanations.
- [ ] `last-updated` bumped when changes affect agent behavior.
- [ ] Related skills and references are listed so agents can navigate without searching.
- [ ] `just lint` passes, confirming valid frontmatter.
