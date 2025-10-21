Status: Implemented (scope: project management review snapshot; caveats: reflects repository state on 2025-10-20)

# Review 12 — Project Management (Agents + Owner)

Provenance
- Topic: 12 — Project management (agents and owner)
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 5065563
- Timebox: ~60–90 minutes
- Scope: Coordination mechanisms: VK workflows, status semantics, attempt lifecycle, consolidation from reviews to actions, escalation via Needs-Unblock, cadence; docs/skills only (no code changes)
- Version: v0.1

Context
- Focused review of how agents and the project owner coordinate using VibeKanban (VK), skills, and repo conventions. Goal: surface gaps, ambiguities, and opportunities to tighten the day-to-day loop and review-to-action flow.

Inputs & Method
- Skills read: `skills/vibekanban.md`, `skills/collaboration-reporting.md`, `skills/roles-scope.md`, `skills/repo-onboarding.md`, `skills/testing-and-ci.md`, `skills/good-code-loop.md`.
- Owner workflow doc: `docs/workflows-project-owner-vibekanban.md`.
- Commands run (outcomes):
  - `uv run python scripts/load_skills_metadata.py` — OK (skills summaries printed)
  - `uv run mkdocs build --strict` — OK (strict build clean)
  - `git status -sb` — clean except for this review’s new file
  - `rg --files` / targeted `sed -n` reads — OK
  - VibeKanban tools (MCP): list projects/tasks — OK (project “Msc Math Viterbo”; task Topic 12 present, in-progress)

Findings (unsorted)
- VK is the canonical backlog; skills reinforce “keep ticket text lean; encode state via columns,” reducing duplication of status prose in descriptions.
- Escalation via `Needs-Unblock: <topic>` is consistently documented across skills and AGENTS.md, making blockers visible without inventing per-task tags.
- Governance is explicit: Owner (Jörn) merges and owns scope/CI; Advisor (Kai) influences research direction but does not gate merges. This separation reduces merge latency and confusion.
- Weekly reporting cadence exists in `skills/collaboration-reporting.md`; it prescribes concise summaries with artefacts archived under `mail/` and `artefacts/`, keeping the repo clean.
- Attempt lifecycle is partially exposed in tools: fields `has_in_progress_attempt`, `has_merged_attempt`, `last_attempt_failed` exist on tasks; a `start_task_attempt` endpoint requires `base_branch` and an `executor` choice.
- Executors enumerated in tools (`CLAUDE_CODE`, `CODEX`, `GEMINI`, `CURSOR`, `OPENCODE`) imply multi-agent capability; recommend documenting when to choose which and the acceptance gates per executor.
- Work execution happens in per-task git worktrees under `/var/tmp/vibe-kanban/worktrees/<task-slug-or-id>`; this isolates changes and reduces branch clutter.
- Owner workflow doc provides operational steps (bring-up, tunnels, VK server, MCP, Cloudflare exposure) so agents can assume a stable environment.
- VK renderer quirks are known: underscores can create italics; the skills prescribe backticks/escapes and Unicode math over LaTeX for reliability.
- A Cloudflare Worker-based sanitizer exists to protect outbound VK text (underscores), indicating attention to text hygiene and cross-tool consistency.
- Status semantics: tools mention status filter values like `todo`, `inprogress`, `inreview`, `done`, `cancelled`; task listings display `in-progress` (hyphenated). This discrepancy should be normalized in docs/tools to avoid filter bugs.
- Board columns are the single source of state; skills explicitly discourage duplicating status strings inside descriptions, which keeps tickets skim-friendly.
- Task updates are encouraged in comments/description edits with pointers to artefacts; this keeps VK as the live log without bloating the main description.
- “Keywords:” optional line for searchability is a good pattern but should be time-bound to avoid drift.
- Orchestrator prompt (docs/prompts/orchestrator.md) defines a lightweight operational loop: backlog check, decide actions, prep handoffs, and broadcast summaries to Owner.
- Consolidation concept is present in the backlog (“Consolidate: Publish review index links + provenance tidy-up”), signaling a downstream step that translates reviews to actions and housekeeping.
- Reviews follow an unsorted, long-list pattern with provenance; the template and Topic 08 establish a consistent review format, which improves comparability and reuse.
- Reviews are not in MkDocs nav by design; the index (reviews/README.md) lists them. This keeps nav compact but relies on authors to add links (later consolidated per current task brief).
- Pre-submit checks are explicit in multiple places: `uv run mkdocs build --strict` and optionally `just checks`; this gate reduces broken-link churn.
- “Golden commands” and use of `uv run` provide a stable Python environment and common entry points, which helps agents avoid local drift.
- The Owner workflow recommends persisting VK data and caches across container rebuilds; this stability supports long-running, multi-attempt tasks.
- VK “VK-Safe Formatting” section provides concrete rules (backticks, underscore escapes, fenced LaTeX) that agents can copy-paste into updates to avoid rendering regressions.
- Attempt naming/branching conventions are not documented in skills; consider standardizing branch names like `task/<id>-<slug>` to improve grepability and cleanup.
- No explicit “Attempt → Review → Merge” state machine is described; tools and skills imply it but do not diagram the transitions and gates (e.g., required reviews, CI states).
- Tasks show `last_attempt_failed`; policy for how many retries and when to escalate is not documented; add thresholds (e.g., escalate on 2 consecutive failures).
- VK-to-repo linkage: there’s no field binding a ticket to a primary doc path; conventionally, reviews and briefs include filenames. A formal field would improve traceability.
- “Consolidation from reviews to actions” lacks a defined recipe: who creates follow-up tasks, when, and what acceptance criteria they must include.
- The repo encourages small, incremental steps; still, it lacks a playbook for breaking large review outputs into bite-sized, testable tasks with owners and due windows.
- Needs-Unblock taxonomy is ad hoc (`architecture`, `communication`, `devcontainer`, `onboarding`, etc.) but effective; consider cataloging common tags to avoid synonyms.
- Skills emphasize “do not revert files you didn’t edit,” reducing merge conflicts and accidental churn from agents.
- CI parity (`just ci`) is a recommended gate pre-handoff; making it a hard requirement for code changes, and optional for docs-only, would remove ambiguity.
- Weekly cadence: skills prescribe summaries and backlog alignment; a “mid-week sync” ritual could improve responsiveness on fast-moving tasks or research pivots.
- Owner doc details VS Code Tunnel + Cloudflared; this supports remote collaboration and easy review without deep local setup.
- VK MCP APIs exist for list/create/update/start-attempt; they could be extended for bulk operations (migration, mass retagging) per Owner doc suggestions.
- Renderer caveats are addressed upstream (recommendations to VK maintainers) and downstream (sanitizer), showing a two-pronged strategy for stability.
- Agents are asked to capture exact commands and outputs when failures occur; encouraging short redacted logs (≤10 lines) in VK comments will reduce rework.
- There’s no explicit “Change freeze” or “CI maintenance window” policy; adding light governance would reduce surprises during heavy maintenance.
- The “orchestrator role” is documented; a formal rotation or checklist could help when multiple people share orchestration responsibilities.
- The board shows several topics in-progress and done; the presence of “has_in_progress_attempt” aligns with the live worktree model and facilitates triage.
- Task priorities (prio:1/2/3) aren’t formalized in skills; a simple convention (e.g., `Keywords: prio:2`) plus column order could help stack-ranking.
- VK-safe markdown advice mentions Unicode math; this reduces dependency on KaTeX/MathJax in VK.
- Owner doc mentions uv hardlink warnings; treating them as expected avoids time sink troubleshooting.
- Security hygiene (tokens persisted via volumes, not images) is documented; this reduces accidental credential leaks during agent work.
- The repo keeps reviews under `docs/reviews/` and briefs under `docs/briefs/`; the distinction is clear in intent (observations vs proposals) but could be reiterated in the index.
- The reviews index has a pre-submit checklist; current task brief temporarily disables adding index links to avoid conflicts, which is consistent with a later consolidation step.
- “Actions extracted” is called out in multiple review docs; this enforces at least a minimal handoff payload for planning.
- No explicit SLA for Owner responses to `Needs-Unblock` is documented; setting expectations (e.g., 24–48 hours) would reduce anxiety and idle time.
- Attempt executors differ in capability; documenting known strengths/limits (e.g., code editing vs planning-heavy tasks) would sharpen routing.
- VK tickets sometimes include “Keywords”/metadata lines; a linting pass (regex rules) could keep them tidy and removable when stale.
- The backlog shows “Future:” tasks; documenting how these are triaged into estimable work would keep the board from becoming a parking lot.
- “Open questions for the Owner” in the Owner doc is a good pattern; replicate this as a standard footer in complex tickets to focus responses.
- The Owner doc’s suggestion to add a “Copy worktree path” button in VK is a minor UX win that would accelerate agent onboarding to a task.
- “Start attempt” should capture provenance (executor, base, timestamp) in the ticket automatically; if not, document the manual note format.
- Cross-linking skills in task notes is encouraged; adding a short “Related skills” section in tickets could help new agents faster.
- Documentation explicitly instructs not to add MKDocs nav links for reviews now; consolidations publish links later. This reduces merge conflicts during heavy review bursts.
- The `mkdocs.yml` strict mode is enabled; keeping it green before merges preserves docs quality during rapid iteration.
- The orchestration prompt recommends starting from clean `main`; enforcing this before spinning a worktree reduces merge headaches and accidental drift.

Actions (first pass)
- Normalize VK status semantics: document and enforce one representation (`in_progress`/`inprogress` or `in-progress`) across tools, skills, and filters.
- Document an explicit Attempt Lifecycle: states (planned → in_progress → in_review → merged/failed), gates (CI green, review check), and escalation after N failures.
- Add a short “From Review to Actions” recipe in `skills/vibekanban.md`: who creates follow-up tasks, expected acceptance criteria, and timing.
- Create a canonical `Needs-Unblock` tag list (architecture, communication, devcontainer, onboarding, repo-layout, CI) and encourage reuse; add examples.
- Add a “Cadence” block to `skills/collaboration-reporting.md`: daily VK updates, weekly summary, optional mid-week sync; include SLA expectation for Owner responses.
- Introduce branch naming for attempts: `task/<id>-<slug>` and require linking attempt branch in the VK ticket when starting.
- Extend MCP tools/docs with “reference doc path(s)” per ticket to hard-link reviews/briefs and improve traceability.

Cross-References
- Topics: T08 (Review process), T11 (Project owner workflows)
- Files: `skills/vibekanban.md:1`, `skills/collaboration-reporting.md:1`, `skills/roles-scope.md:1`, `skills/repo-onboarding.md:1`, `skills/testing-and-ci.md:1`, `docs/workflows-project-owner-vibekanban.md:1`, `docs/reviews/README.md:1`, `mkdocs.yml:1`

Validation
- `uv run python scripts/load_skills_metadata.py` — OK (skills loaded)
- `uv run mkdocs build --strict` — OK (site builds, strict)
- VK MCP list projects/tasks — OK (project located; Topic 12 in-progress)

Limitations
- Did not run `just ci`; changes are docs-only. Did not test VK UI directly; relied on skills/tools and Owner doc. Attempt lifecycle suggestions based on available fields and tools.

Status
- Finalized
