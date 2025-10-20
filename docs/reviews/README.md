---
title: Reviews Index
---

# Reviews — Index and Conventions

Purpose
- Central index for all repository reviews. Each review focuses on a single topic to produce a long, unfiltered, unsorted list of findings. Results feed into consolidation and planning.

How to add a review
- Use the template: `docs/reviews/TEMPLATE.md` (copy and fill the placeholders).
- Create: `docs/reviews/XX-<stub>.md` where `XX` is a zero‑padded topic number.
- Content: be thorough; unsorted is fine; redundancy OK. Use clear, specific language. No length limit.
- Deliverable path: keep the file under `docs/reviews/` so it publishes and is easy to reference.
- Cross‑link: add a single bullet link here after saving the file.

Pre‑Submit Checklist (per review)
- Linked from this index under Published reviews
- `uv run mkdocs build --strict` green
- `just checks` green (or N/A for docs‑only changes)
- Actions extracted (even minimal is fine)
- Cross‑refs noted (topics/files)
- Provenance filled (Topic, Author, Date, Commit, Timebox, Scope, Version)
- Title follows `Review XX — <Topic Title>`

Numbering (topics)
1. Project Design Principles
2. Onboarding for new agents
3. Repo and code architecture
4. Code
5. Developer documentation
6. Math used
7. Algorithms used
8. Review process and reviews in general
9. Advisor‑facing documentation/presentation
10. MSc thesis
11. Project owner workflows
12. Project management (agents and owner)
13. Repo readiness for AI vs humans
14. Misc

Published reviews
- 01 — Project Design Principles: 01-project-design-principles.md
- 02 — Onboarding for New Agents: 02-onboarding.md
- 03 — Repo and code architecture: 03-repo-architecture.md
- 04 — Code: 04-code.md
- 05 — Developer documentation: 05-developer-documentation.md
- 06 — Math used: 06-math-used.md
- 07 — Algorithms used: 07-algorithms-used.md
- 08 — Review process and reviews in general: 08-review-process.md
- 09 — Advisor‑facing documentation/presentation: 09-advisor-docs.md
- 10 — MSc thesis: 10-msc-thesis.md
- 11 — Project owner workflows: 11-owner-workflows.md
- 12 — Project management (agents and owner): 12-project-management.md
- 13 — Repo readiness for AI vs humans: 13-repo-readiness-ai-vs-humans.md
- 14 — Misc: 14-misc.md

Upcoming
