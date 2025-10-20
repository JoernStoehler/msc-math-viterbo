# Review 02 — Onboarding for New Agents

Provenance
- Topic: 2 — Onboarding for new agents
- Author: Agent (Codex CLI)
- Date: 2025-10-20
- Commit: 96e7f5acdfb35cd1e3e5030d9f27aafa3f712db3
- Timebox: ~90 minutes
- Scope: docs, skills, golden commands, devcontainer flows; focus on time-to-first-green checks, skill discoverability, minimal read path
- Version: v0.1

Context
- Review objective: reduce time-to-first-green checks and ambiguity for new agents. Emphasis on minimal reading path, discoverability of the right skills, clarity of golden commands, and container/startup flows.

Inputs & Method
- Commands run: `uv run python scripts/load_skills_metadata.py --quiet --check`, `uv run mkdocs build --strict`, targeted `rg` queries, light reads of skills, Justfile, devcontainer docs.
- Files consulted (representative): `AGENTS.md`, `skills/*.md`, `skills.index.md`, `Justfile`, `mkdocs.yml`, `docs/charter.md`, `docs/reviews/README.md`, `.devcontainer/README.md`, `.devcontainer/devcontainer.json`, `.devcontainer/bin/*`, `scripts/inc_select.py`.

Findings (unsorted)
- The minimal read path is explicit and short: AGENTS.md Boot Sequence instructs to (1) read the task on VibeKanban, (2) run the skills metadata loader, (3) open relevant skills via the map; this strongly favors fast orientation over broad reading.
- AGENTS.md’s Always-On Essentials concisely state the golden commands (`just checks`, `just test`, `just ci`, `just fix`, `just bench`) and shell practices (`rg`, ≤250 lines), which reduces decision overhead for new agents.
- Skills are discoverable from two entry points: (a) AGENTS.md’s “Skill Map” grouping, and (b) `skills.index.md` task-driven clusters; both are short and actionable.
- Skills metadata is auto-generated into AGENTS.md; `scripts/load_skills_metadata.py` prints, checks, and can fix the AGENTS.md blocks. This helps keep the index fresh without manual churn.
- Running `uv run python scripts/load_skills_metadata.py --quiet --check` emits a warning that Always-On is empty, which appears intentional (no skills marked `relevance: always`). The warning is low-noise and doesn’t block onboarding.
- The charter explicitly sets an onboarding SLO: median time to first green `just checks` ≤ 10 minutes in a fresh daily worktree, which sets a measurable expectation for onboarding performance.
- Golden commands are well-implemented in the Justfile. `checks` runs format → lint → type → test (incremental), aligned with fast feedback loops.
- Incremental test selection is implemented via `scripts/inc_select.py` and leveraged by `just test` and `_pytest-incremental`. The selector is graph-based (AST import graph + file hashes) with conservative fallbacks, improving TTFB for tests.
- The selector gracefully skips pytest when there are no Python changes and no prior failures, saving minutes in a typical documentation-only or planning cycle.
- The test suite demarcates tiers via markers (`smoke`, `deep`, `longhaul`) and keeps smoke lean; many heavy tests are skipped pending backends, which keeps early runs fast for newcomers.
- MkDocs is a first-class citizen: strict build is part of `just ci` and a single `docs-build` recipe exists. This makes doc changes verifiable by a single command and guards against link drift.
- The repository includes a reviews system with an index and a template; onboarding review writers get a clear scaffold (provenance, inputs, validations), minimizing meta-work.
- The README Quickstart foregrounds uv and Justfile commands; it avoids deep explanations and points back to AGENTS.md and architecture docs for details — a good split between “do now” and “read later.”
- Devcontainer flows are documented for the project owner in depth, with host-admin and container-admin scripts. This is overkill for most new agents but crucial for the owner and makes the environment reproducible.
- .devcontainer `post-create.sh` and `post-start.sh` exist to bootstrap uv and print diagnostics, reducing manual setup inside the container.
- Devcontainer JSON uses JSON with comments (JSONC style) in `mounts`; VS Code devcontainers support this. Bind mounts are clearly documented to persist caches and VibeKanban data, directly benefitting onboarding speed (shared uv cache, saved worktrees).
- The Justfile sets `UV := env_var_or_default("UV", "uv")`; users can override uv binary location via `UV` env var if needed; defaults keep commands predictable.
- Torch is a heavy dependency; the Justfile and architecture docs call out techniques to keep CPU-only wheels and speed up CI (`ci-cpu`), and uv caches are mounted in the devcontainer. First-ever sync on a brand-new machine can still be slow; caching strategy mitigates this in intended environments.
- The repo enforces Pyright (basic) across `src/viterbo` and keeps a strict config separately. This surfaces typing issues quickly without forcing strictness for all modules.
- The Ruff setup prioritizes correctness and policy checks over cosmetic style and is pre-configured for the typical directories, aligning with low-friction onboarding.
- The skills/good-code-loop.md file lays out planning and PR hygiene in straightforward terms; new agents can adopt it without reading multiple documents.
- Navigation and file layout in README are concise and accurate; the architecture overview exists in both AGENTS.md (daily) and docs (reference), which aids orientation and depth-on-demand.
- The VibeKanban skill gives concrete board usage rules (escalations via `Needs-Unblock`, formatting tips for underscores, renderer caveats), which are onboarding-relevant in practice.
- The owner workflow doc for VibeKanban is detailed, covering Cloudflared, Code Tunnel, and host setup. It’s clearly targeted at the owner, not every agent, which avoids overwhelming new contributors.
- The repo contains a simple PR template; it’s very light and doesn’t enforce heavy process, which suits fast onboarding but may be too sparse for larger contributions.
- `mkdocs.yml` uses a minimal nav and points to “Reviews: reviews/README.md”; individual reviews aren’t in nav, but the index lists them. Discoverability relies on authors adding the index bullet.
- `docs/charter.md` spells out measurable success metrics (onboarding SLO, CI p95, coverage floors, docs freshness), which anchor expectations for new agents and reviewers.
- Tests under `tests/` include a smoke test file and clearly marked deep/performance suites; many optional/perf tests skip if tooling isn’t installed. This helps keep `just checks` quick.
- The incremental pytest selector captures prior failing nodeids via `.cache/last-junit.xml` to re-run failures even if there are no new changes, improving feedback on fixes.
- The Justfile has system-Python variants for CI-like runs (`ci-cpu`, `test-sys`), which can be useful for onboarding in constrained environments without relying on uv run every time.
- The repo documents clear layering boundaries in `docs/architecture/overview.md` and skills; new agents get strong guardrails (e.g., math purity, no I/O/state) before touching code.
- The `skills/repo-onboarding.md` skill is explicit about the sequence (AGENTS.md → skills loader → relevant skills) and contains a quick command reference, which matches the Boot Sequence and reduces cognitive load.
- Skills are consistently marked with frontmatter (`name`, `description`, `last-updated`), making the generated overview useful and trustworthy.
- The AGENTS.md includes Safety reminders (device/dtype explicit; do not revert files you didn’t edit), preventing common onboarding hazards.
- The repository ships `.editorconfig`, `pyrightconfig.json`, `pytest.ini`, and a coherent pyproject; the environment is explicit and predictable for new agents.
- The Justfile includes `docs-serve` for local preview and `docs-check-links` as `docs-build`, making documentation work ergonomic.
- The reviews system encourages an unsorted, unfiltered list; redundancy is acceptable. This lowers friction for authors and aligns with the “fast to write, easy to scan” philosophy for onboarding.
- `skills/testing-and-ci.md` exists and is referenced from the skill clusters; it centralizes testing posture and CI parity, reducing the need to infer workflows.
- The environment doc (`docs/environment.md`) is owner-focused but doubles as a precise reference for anyone encountering drift; it notes uv hardlink nuances and acceptable slow paths.
- The devcontainer bin scripts (`host-admin`, `container-admin`) integrate status, preflight checks, `code tunnel`, Cloudflare, and VibeKanban orchestration. They expose sensible attach/status behaviors and TMUX hints for multi-service sessions.
- The plan/update cadence for agents is baked into AGENTS.md and skills; work should be structured with small steps and incremental updates, supporting clear onboarding expectations.
- The repo is disciplined about strict docs builds and a light MkDocs config (Material theme), making documentation approachable and less fragile for new contributors.
- Many heavy algorithms in tests are guarded with skips or minimal parametrize, avoiding early flakiness or long runs; this is friendly to new environments.
- The repo avoids committing large artefacts and provides directories for mail/artefacts; onboarding agents get clear separation of code vs artefacts.
- The `uv.lock` is present, signaling reproducible dependency resolution; combined with uv cache and per-worktree .venv, onboarding loops are minimized after the first sync.
- The Justfile prominently features a `help` target and prints `just --list`; this is a gentle, discoverable way for new agents to self-serve.
- The pre-submit checklist in `docs/reviews/README.md` enforces mkdocs strict build green and the index link, aligning with a “no surprises” onboarding flow for docs-only changes.
- The repo defaults to CPU baselines and avoids CUDA complexity by default, which reduces setup pitfalls for newcomers.
- The math code is Torch-first and keeps tensors central, which new agents with PyTorch experience can navigate quickly.
- Skill files are short (1–5 minutes to read each) and specific; this encourages “load only what you need,” a key onboarding optimization.
- The AGENTS.md skill overview includes last-updated timestamps for quick freshness checks; this builds trust in the guidance.
- The repo clearly distinguishes reference architecture docs vs everyday AGENTS.md guidance; this avoids conflating deep rationale with daily usage.
- Worktrees are used per task; the environment is designed to run Codex agents per worktree. This isolates changes and prevents accidental cross-task interference, a subtle but strong onboarding aid.
- The reviews and briefs structures coexist; reviews are for observations, briefs for plans. The index and template make this discoverable to new agents.
- Ownership is named (Jörn; advisor Kai), and escalation tags are standardized (`Needs-Unblock: <topic>`), making governance clear from day one.
- The “PDF to Markdown (quick)” tip in AGENTS.md is pragmatic and placed where agents will actually see it — helpful for onboarding documentation hygiene.
- The repo does not require immediate C++ toolchain setup to get green checks; C++ extensions have Python fallbacks and optional benches are skipped without plugins.
- The `docs/charter.md` duplicates one acceptance criterion in “Atlas Tiny” by mistake (listed twice) — minor nit; not onboarding-critical but noted.
- MkDocs build reports “pages exist but are not included in nav” (reviews, briefs, papers). This is acceptable and documented; strict mode does not treat this as an error, so onboarding is not blocked.
- The `.devcontainer/README.md` and owner runbook clearly state “No Codespaces; no Codex Cloud” for the owner’s golden path; new agents joining outside that path should still succeed via README+AGENTS+Justfile using local uv.
- The Justfile explicitly provides `docs-build` and `docs-serve` which match the onboarding task in this review; command names are consistent and memorable.
- The test suite sets `SMOKE_TEST_TIMEOUT` and tunes selection logic; onboarding agents can set `INC_ARGS="--debug"` to understand impacted selection — helpful in early learning.
- The repo keeps notebook usage minimal and optional; onboarding does not require Jupyter to get to green checks.
- Skills/collaboration-reporting.md reminds authors to cross-link new skills/docs from AGENTS.md so the onboarding funnel stays in sync.
- The repository’s default Python is 3.12, set in pyproject, and the devcontainer image is compatible; version mismatches are unlikely for new agents in the intended environment.

Actions (first pass)
- Add one “always-on” skill entry (e.g., `basic-environment`) by marking `relevance: always` to remove the recurring Always-On warning during metadata checks (optional; low priority).
- Add a line in README Quickstart suggesting `just docs-build` when making docs-only changes; keeps the mental model consistent with “golden commands”.
- In `docs/charter.md`, fix the duplicated acceptance criteria bullet under “Atlas Tiny demo” for polish (non-blocking).
- Add a short “New Agent: 5-minute path” note in AGENTS.md (e.g., AGENTS.md: Boot Sequence) explicitly stating: AGENTS.md → skills loader → Justfile `checks` → read only 2 skills (`repo-onboarding`, `basic-environment`) before first change.
- Consider adding `just docs-serve` to README as an explicit docs iteration path for onboarding reviewers.
- Optional: add a `just setup` pointer in AGENTS.md’s Boot Sequence step 2 to run the fix mode (`scripts/load_skills_metadata.py --fix`) when metadata drift is detected.

Cross-References
- AGENTS.md:1
- skills/repo-onboarding.md:1
- skills/basic-environment.md:1
- skills/testing-and-ci.md:1
- skills.index.md:1
- Justfile:1
- mkdocs.yml:1
- docs/charter.md:1
- .devcontainer/README.md:1
- .devcontainer/devcontainer.json:1
- .devcontainer/bin/host-admin:1
- .devcontainer/bin/container-admin:1
- scripts/inc_select.py:1

Validation
- Skills check: `uv run python scripts/load_skills_metadata.py --quiet --check` — OK (warning: Always-On empty; acceptable)
- Docs: `uv run mkdocs build --strict` — OK (pages not in nav are informational)

Limitations
- Did not run `just checks` end-to-end to avoid heavyweight dependency install in this context; relied on test layout and selector inspection to assess TTF green. In the target environment (devcontainer + uv cache), first green should meet the charter SLO.
- Devcontainer flows reviewed at the documentation level; did not bring up tunnels or VibeKanban in this run.

Status
- Finalized

