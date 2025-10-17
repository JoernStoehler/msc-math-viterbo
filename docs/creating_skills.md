---
title: Creating Skills
---

# Creating Skills

Use the `skills/` directory to capture agent-facing workflows once they stabilize beyond a single task. Each skill is a Markdown file with YAML frontmatter that keeps metadata lightweight while allowing future conversion to Anthropic's `SKILL.md` format.

## When to Add or Update a Skill

- Consolidate correlated workflows into one file (e.g., devcontainer lifecycle + CLI cheats) rather than scattering instructions.
- Split only when two audiences rarely need both workflows in the same task.
- Update the `last-updated` field whenever a meaningfully new policy lands; note the change in the file body.

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
- Body: focus on actionable procedures; push large references to `docs/` and link back.

## First-Action Script

- After reading `AGENTS.md`, run `uv run python scripts/load_skills_metadata.py` to stream metadata into the model context.
- The script prints one summarized line per skill; no instructions are emitted, keeping the context budget small.
- If a skill is critical for every task, link it from `AGENTS.md` so agents load it explicitly after running the script.

## Review Checklist

- [ ] Metadata describes when the skill applies (not just what it contains).
- [ ] Instructions use imperative voice and assume the reader is another Claude agent.
- [ ] Links point to canonical docs/scripts; avoid duplicating long-form explanations.
- [ ] `last-updated` bumped when changes affect agent behavior.
