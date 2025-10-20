# Review 11 — Project Owner Workflows

Provenance
- Topic: 11 — Project Owner Workflows (VK cadence, escalation, env lifecycle, backups, release/archival, toil)
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 5065563
- Timebox: ~60–90 minutes
- Scope: Owner workflows across VK board usage, lifecycle scripts, persistence/backups, release/archival habits, and toil sources; docs-only
- Version: v0.1

Context
- Review the Project Owner’s end-to-end operational workflows (board usage, lifecycle scripts, remote access, persistence) to surface concrete strengths, gaps, and toil. Emphasis on VibeKanban cadence, escalation patterns, environment lifecycle, backups, release/archival steps.

Inputs & Method
- Read: `docs/workflows-project-owner-vibekanban.md`, `docs/environment.md`, `.devcontainer/bin/container-admin`, `.devcontainer/bin/host-admin`, `Justfile`, `mkdocs.yml`, skills under `skills/*.md` (board, environment, CI/testing, collaboration/reporting).
- VK context: listed projects and tasks via VK tools; confirmed task “Review: Project Owner Workflows (Topic 11)” present on project `Msc Math Viterbo`.
- Commands executed:
  - `uv run python scripts/load_skills_metadata.py --quiet --check` (non-mutating metadata/AGENTS checks)
  - `uv sync --extra dev` (ensure mkdocs and dev tools available)
  - `uv run mkdocs build --strict` (docs build parity)

Findings (unsorted)
- Owner workflow is explicitly documented end-to-end with concrete commands for host and container, reducing ambiguity for daily bring-up and remote access.
- Lifecycle is intentionally decomposed (independent services for VK UI, Code Tunnel, Cloudflared) which isolates failures and allows targeted restarts without disrupting everything.
- Hot-fix path to restart only VibeKanban exists and is emphasized as safe; this directly addresses common UI-state corruption without impacting tunnels.
- Named Cloudflared tunnel with bind-mounted creds provides durable, host-backed identity; supports predictable DNS routing and scoped Access policy.
- Public VK access is protected via Cloudflare Access; this models clear responsibility boundaries for the Owner (Zero Trust policy) vs agents (operate inside devcontainer).
- Data persistence strategy is bind-mounted and simple: VK app data and worktrees live on host directories, avoiding Docker volume indirection and simplifying backups.
- VK binaries are not forked; UI tweaks (font, sanitizer) are injected at the edge (Cloudflare Worker), keeping upstream pristine and upgrades low-risk.
- The VK text sanitizer addresses a known pain (intraword underscore italics) with a minimal allowlist proxy; hygiene guidance is duplicated in skills to avoid regressions.
- Lifecycle scripts provide idempotent starts and best‑effort stops with readable logs, encouraging operators to rely on scripts over ad hoc process management.
- Preflight checks surface missing dependencies (npx, code tunnel, cloudflared, tunnel config) early with actionable hints, reducing confusing half-start states.
- tmux integration in detached start gives convenient multi-window layout and repeatable session management; discoverability tips are printed in interactive mode.
- Owner golden-path explicitly rejects Codespaces/Codex Cloud, favoring a local host + devcontainer; clarity avoids split-brain and inconsistent environments.
- uv cache vs .venv filesystem mismatch is accepted with a small copy penalty; guidance is explicit so operators do not chase phantom perf issues.
- Device/dtype policies and stack versioning live elsewhere (skills/docs) but are referenced sufficiently for owner-level operations; avoids policy drift during ops.
- VK board is the single backlog, and tickets encode status via columns; description/comments remain lean and link out to docs/skills; reduces duplication of state.
- Escalation convention is lightweight and unambiguous (`Needs-Unblock: <topic>` in the ticket body) and is repeated across skills to make it visible to all agents.
- Weekly reporting cadence exists (collaboration skill), but the owner workflow doc does not embed a “when to groom/triage” schedule; cadence is implicit, not measured.
- No explicit SLOs for board hygiene (e.g., max age of in-progress without updates); potential for drift if not reinforced in reviews/checklists.
- Host one-shot orchestration (`host-admin up preflight start --interactive`) makes daily bring-up reliable; mirrors the in-container `container-admin start` for fallback.
- Owner release process is deliberately non-PyPI: publish is disabled; release command bumps version and tags, aligning with internal milestones rather than packaging.
- Archival intent is stated in the Charter (end of March 2026) but lacks a concrete, step-by-step archival playbook (site freeze, dataset artefact snapshot, Access teardown).
- Backups are “easy” by design (host bind mounts), but there’s no documented schedule/tooling (e.g., systemd timers, rclone targets, retention policy, test restores).
- Worktrees live under `/var/tmp/vibe-kanban/worktrees` (bind-mounted); robust against container rebuilds, but there’s no routine to prune stale worktrees or enforce quotas.
- Cloudflare credentials and config live under `~/.cloudflared/` (bind-mount), but documentation does not include a credential rotation procedure or incident drill.
- VS Code Tunnel dependency introduces friction when CLI updates change flags; scripts attempt to install, but recovery guidance is limited to log inspection.
- Node/npm reliance for VK UI and Wrangler deploys is explicit; scripts handle installs, but version pinning and update cadence are not specified, risk of surprise upgrades.
- The VK “Open in VS Code” UX is intentionally de-emphasized in favor of CLI `code --add`; avoids coupling to upstream, but sacrifices a small QoL.
- Owner’s “fast loop” expectations exist (quick bring-up, small penalties accepted); however, there’s no quantified cold-start target for owner daily start.
- Cloudflare Worker deployment is a manual step unless `CF_AUTO_DEPLOY=1` is set; explicit but easy to forget; no automatic check of Worker versions post-deploy.
- Access allowlist management is mentioned but not proceduralized (no named groups or rotation cadence); potential toil during collaborator churn.
- VK formatting rules are duplicated (skill + workflow); good for agent consistency, but could diverge without a single source of truth marker.
- Skills index auto-generation is wired, but currently warns about “always-on skills” being empty; not a blocker, but creates recurring low-grade noise if left unresolved.
- Owner flow assumes a stable Ubuntu host; cross-host portability (e.g., macOS) is not a goal; docs do not capture divergent steps for alternate hosts.
- `host-admin` protects against misuse (guard clauses for host vs container), which is important for novice operators interacting from mixed shells.
- `container-admin` and `host-admin` both surface status checks; standardization of exit codes for higher-level orchestrations is not defined (but readable logs help).
- There is no explicit disaster-recovery drill (e.g., how to rebuild from only host backups and a fresh repo clone); “one-shot” scripts cover happy path.
- VK data path and presence of `db.sqlite` are documented; a checksum or periodic export routine (sqlite dump) is not documented; restores remain untested.
- Artefact handling is clear (keep heavy outputs in `artefacts/`); but owner workflow does not spell out a rotation/retention policy for large artefacts.
- Email hygiene for advisor/maintainer is codified; aligns with keeping sensitive attachments out of repo; ensures owner avoids accidental exposure.
- CI parity (`just ci`) includes a strict docs build, which keeps owner-facing docs fresh; helpful to catch drift introduced by doc-only tickets.
- No explicit “owner dashboards” for health metrics (uptime, tunnel status, worker status); operators rely on commands and logs rather than a single pane of glass.
- Named tunnel config helper writes a minimal cloudflared YAML; update strategy for hostname/port changes is “re-run helper” rather than diff-aware edits.
- Ownership and escalation roles are clear (Owner: Jörn; Advisor: Kai); non-owner agents are guided to escalate through VK tickets rather than PRs.
- `Justfile` release target bumps version and tags but does not create release notes or signed tags; for thesis milestones, this may be sufficient but lacks narrative.
- Docs build warns about pages outside nav (expected); owner workflow accepts that reviews/briefs live without nav links until consolidation; reduces merge conflicts.
- Rebuild scripts support `--no-cache`; encourages clean rebuilds when base image changes; fine-grained cleanup (dangling containers/images) is not integrated.
- VS Code Tunnel start/attach paths are well documented; state drift (e.g., stale tokens) is mentioned in environment notes but not scripted into diagnostics.
- No automated check that VK UI is reachable via Access URL post-start; status is manual (logs, curl); a simple HTTP probe could reduce guesswork.
- Cloudflare Workers have separate wrangler configs; consolidated deploy helper exists; lack of version pinning or rollout notes is a minor operational risk.
- Board usage emphasizes concise tickets with links; explicit “reference doc path(s)” field is suggested as an improvement to reduce search toil.
- Ticket templates in VK are not enforced; suggestion exists to add structured fields (why, acceptance, test steps, refs) to reduce inconsistent ticket quality.
- Owner “Answers to current questions” section provides rationale for policy choices; this improves bus factor for future operators.
- Network dependencies (npm registry, Cloudflare APIs, VS Code updates) are acceptable at owner-level but are not mirrored/offline-capable; acceptable risk trade-off noted.
- Devcontainer start scripts install missing CLIs (just, cloudflared) best-effort; partial installs are possible when network is constrained; recovery advice is minimal.
- `uv sync` and dependency lockfiles stabilize Python side; Node and tunnel tooling do not have equivalent lock/pin mechanisms documented; potential churn.
- `mkdocs strict` in CI means any broken links/warnings would gate merges; this nudges owner to keep operational docs up-to-date alongside changes.
- Thesis archival end-state lacks explicit instructions for freezing the doc site (static export) and capturing a final artefact manifest; implied but not codified.
- No monitoring for storage quotas on bind-mounted dirs; large `artefacts/` or VK sqlite growth could surprise the host; manual hygiene required.
- The board currently hosts many review tickets concurrently as in-progress; an owner cadence to timebox and serialize reviews might reduce context-switching.
- There’s no defined “owner on-call” rotation (single owner); escalation paths rely on the same person, which is okay for a solo project but noted as a single-point risk.
- The sanitizer Worker alter outbound JSON under `/api/*`; very targeted but still an operational component; no roll-back SOP besides redeploying an older config.
- Owner flow calls out quick tunnels vs named tunnels; quick tunnels are convenient for dev but not scripted; named tunnel is the canonical choice.
- The repo has a clean separation between owner ops docs and implementation; clarity reduces bikeshedding in tickets that should focus on math/code.

Actions (first pass)
- Add a short archival playbook under `docs/` covering: final tag + site freeze, dataset/artefact manifest, VK db export, Access teardown checklist.
- Add a simple backup section (owner docs) with recommended command snippets (rsync/rclone) and a weekly retention suggestion; include a test-restore note.
- Add a `container-admin probe` that curls the local VK URL and prints Access status hints; use in `start` to fail early if UI is not reachable.
- Pin minimal Node/npm ranges and document a npx fallback policy; add a tip to clear npm cache when VK fails to start after upstream updates.
- Document a quarterly credential rotation SOP for Cloudflare tunnel and Workers; include the exact files/commands impacted by rotation.
- Add an optional `owner-health.md` “dashboard” page listing one-liners for status of VK, tunnel, Workers, VS Code tunnel; link logs paths.

Cross-References
- Topics: T01 (Design Principles), T02 (Onboarding), T03 (Repo Architecture), T08 (Review Process), T12 (Project Management).
- Files: `docs/workflows-project-owner-vibekanban.md:1`, `docs/environment.md:1`, `.devcontainer/bin/container-admin:1`, `.devcontainer/bin/host-admin:1`, `Justfile:169`, `mkdocs.yml:1`, `skills/vibekanban.md:1`, `skills/devcontainer-ops.md:1`, `skills/collaboration-reporting.md:1`.

Validation
- `uv run python scripts/load_skills_metadata.py --quiet --check` — OK (warn: always-on-skills empty)
- `uv sync --extra dev` — OK
- `uv run mkdocs build --strict` — OK (built site; pages outside nav as expected)
- VibeKanban tools — Listed projects and found task “Review: Project Owner Workflows (Topic 11)” under `Msc Math Viterbo`.

Limitations
- Did not run owner lifecycle scripts end-to-end (network/external services mocked by reading scripts and docs).
- Backup/restore observations are derived from documented bind mounts; no destructive restore drills performed.
- Release/archival steps inferred from Charter and Justfile; no owner-specific archival doc currently exists to validate.

Status
- Draft

Pre-Submit Checklist
- Do not add nav/index links for this review (consolidation will publish later)
- `uv run mkdocs build --strict` green
- `just checks` N/A (docs-only)
- Actions extracted (initial pass)
- Cross-refs noted (topics/files)
- Provenance filled
- Title follows `Review XX — <Topic Title>`
