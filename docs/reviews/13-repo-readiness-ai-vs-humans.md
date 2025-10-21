Status: Implemented (scope: AI-vs-human readiness review snapshot; caveats: reflects repository state on 2025-10-20)

# Review 13 — Repo Readiness for AI vs Humans

Provenance
- Topic: 13 — Repo readiness for AI vs humans
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 5065563
- Timebox: ~90 minutes (read → draft → validate)
- Scope: Skills coverage, VibeKanban ergonomics, command flows, documentation shape, formatting/safety edge cases
- Version: v0.1

Inputs & Method
- Read `AGENTS.md`, `skills/*.md`, `Justfile`, `mkdocs.yml`, `docs/reviews/README.md`, `.devcontainer/` docs, owner workflow docs.
- Commands: `uv run python scripts/load_skills_metadata.py --quiet`, `rg -n "review|mkdocs|vibekanban|skills|just|uv run|always-on|always on|always_on|AGENTS.md" -S`, `uv run mkdocs build --strict` (post‑write).
- Navigation: used `rg` for fast repo-wide scans; kept shell reads ≤250 lines per file.

Findings (unsorted)
- Skills index exists and is auto‑generated into `AGENTS.md` with clear markers (`AGENTS.md:63` shows the generated block start; the script is `scripts/load_skills_metadata.py`). The “skills-overview” inventory provides name, scope sentence, and last‑updated — good for both agents and humans to triage at a glance.
- No skills are flagged as “always‑on” in the auto‑generated section (`AGENTS.md:90`). For agents, explicitly marking `basic-environment` (and possibly `repo-onboarding`) as `relevance: always` would improve selection defaults and reduce cold‑start overhead.
- Boot sequence for agents is explicit in `AGENTS.md:7` (read task in VibeKanban → run skills metadata loader → open matching skills). This is agent‑friendly and aligns with the VibeKanban launch path.
- Golden commands are concise and repeated in multiple places: `AGENTS.md:18`, `skills/basic-environment.md`, and `Justfile` recipes. This redundancy helps both AI and humans discover “what to run” without digging.
- Strong preference for `uv run` across flows is enforced by skills and Just recipes (e.g., `Justfile:20`, `Justfile:57`). This stabilizes Python entry points for both AI agents and human developers across clean environments.
- The repo ships a robust `Justfile` with clear gates: `checks`, `precommit`, `ci`, `lint`, `type`, `test` variants, `bench`, and docs build (`Justfile:206` includes `mkdocs build --strict`). Command ergonomics are high for humans and easily scriptable for agents.
- Incremental test selection is built-in via `scripts/inc_select.py` and wired in `Justfile:_pytest-incremental` (lines near `Justfile:222`), providing fast inner loops. This benefits both AI and humans; agents can respect INC_ARGS knobs.
- MkDocs site builds from `mkdocs.yml`, with `strict: true` (`mkdocs.yml:7`). This is an excellent guard against broken links and references, and it’s explicitly included in `just ci`. Both AI/humans benefit from a single “docs must build” invariant.
- Reviews index exists and instructs authors to add a bullet under Published reviews (`docs/reviews/README.md:14`, `docs/reviews/README.md:42`). It also provides an explicit pre‑submit checklist useful to both AI and humans.
- Reviews include a TEMPLATE with provenance, inputs/method, findings, actions, cross‑refs, validation, limitations, status, and a final pre‑submit checklist (`docs/reviews/TEMPLATE.md`). This shape is friendly to agents (predictable anchors) while remaining readable for humans.
- Documentation navigation keeps a single “Reviews” entry in nav (`mkdocs.yml:21`), pointing to the index. Individual review pages are discoverable by search and the index; avoiding nav sprawl is a reasonable human ergonomics tradeoff and fine for agents that rely on the index.
- Skills are authored to Anthropic spec with metadata checked by `scripts/load_skills_metadata.py`. The script provides `--check` warnings for out‑of‑date blocks and invalid frontmatter (exit code 2), see `scripts/load_skills_metadata.py:22`. This makes the skills surface safer for agents.
- The skills loader updates `AGENTS.md` auto‑generated sections on `--fix`; the utility detects whitespace‑only differences and bullet set differences (`scripts/load_skills_metadata.py:144` onward). This reduces drift and makes the index a reliable AI entry point.
- `AGENTS.md` explicitly tells agents to “do not revert files you didn’t edit” and to “keep device/dtype choices explicit” (safety guardrails). This is agent‑oriented safety language that also serves as human reviewer expectations.
- Safety note in `AGENTS.md`: “PDF to Markdown (quick)” with `pdftotext` and storing summaries under `mail/private/`. Clear expectations to avoid committing heavy artefacts; this is actionable for both agents and humans.
- Shell output and search ergonomics are optimized for agents: “prefer `rg`” and “stream reads ≤250 lines” are stated in `AGENTS.md` and reflected in the agent tool harness. This consistency improves reproducibility and avoids truncated outputs in UIs.
- Devcontainer: `.devcontainer/devcontainer.json` is explicit (image, post‑create, post‑start, volumes, env). The docs under `.devcontainer/README.md` and `docs/environment.md` provide an unambiguous workflow — humans have clear operational steps; agents can reference it for environment assumptions.
- Devcontainer mounts persist tokens, caches, and worktrees (see `.devcontainer/devcontainer.json:23` mounts list). For human ergonomics, this prevents re‑auth churn; for agents, stable paths like `/var/tmp/vibe-kanban/worktrees` simplify worktree access.
- Owner workflow is fully documented in `docs/workflows-project-owner-vibekanban.md` including Cloudflare Access, named tunnel setup, and expected directory locations. This enables non‑AI stakeholders to operate the system, and gives agents reliable assumptions about URL hostnames and ports.
- VibeKanban integration is described clearly: runs via `npx`, stores data under standard appdata paths, MCP server available, per‑task worktrees created under `/var/tmp/vibe-kanban/worktrees` (docs/workflows-project-owner-vibekanban.md). Agents can rely on the MCP tools; humans can run the UI.
- Command surfaces show consistent terms and tone across docs and skills (e.g., “golden commands”, “smoke/deep/longhaul” markers). This helps agents pattern‑match and helps humans build muscle memory.
- Tests are organized into markers (`smoke`, `deep`, `longhaul`) and categories (`unit`, `integration`) with convenience Just recipes. For agents, marking tests enables targeted validation; for humans, the categories match common expectations.
- PyTorch device/dtype policies are codified in `skills/good-code-loop.md` (float64 math by default; models float32; explicit device semantics). Both AI and human contributors see the contract upfront.
- Architecture guardrails (layering) are documented in `skills/good-code-loop.md` and `docs/architecture/overview.md`. Explicit prohibitions (no upward imports, pure math-layer) help agents avoid class of PR hygiene issues; humans have quick rationale links.
- C++ extension stubs exist with a Python fallback expectation (`docs/architecture/overview.md:27` mentions safe fallback). This gives humans portability while letting agents escalate to C++ only with evidence (`skills/performance-discipline.md`).
- The repo uses Ruff and Pyright with curated rule sets (`pyproject.toml:37` onwards). Lint/type ergonomics are aligned with correctness and policy over cosmetic checks — helpful to agents (fewer false positives) and humans (sane defaults).
- Pre‑commit proxies exist as Just targets (`precommit`, `precommit-fast`, `precommit-slow`) enabling consistent gates before PRs. Agents can invoke them; humans can adopt them without installing pre‑commit hooks.
- CI parity target `just ci` includes docs build (`Justfile:206`). This single-command parity reduces drift between local runs (agent/human) and CI.
- `skills/repo-onboarding.md` prescribes first steps (env, commands, how to read the repo). It’s short and links to deeper skills, which suits both AI agents (token‑budget friendly) and humans (skim first, drill down as needed).
- `skills/skill-authoring.md` is detailed and tailored for AI agents authoring/updating skills. Humans can also use it to review or propose edits to skills; it captures best practices like mixing facts with imperative steps and progressive disclosure.
- The repo explicitly mentions “VibeKanban ergonomics” under `skills/vibekanban.md` with escalation patterns and cadence. For agents, this integrates planning with execution; for humans, it documents how the Kanban board is used day‑to‑day.
- `docs/README.md` points contributors to core areas (Architecture, Math API, Owner workflow). This keeps the landing page simple for humans and gives agents anchor links.
- Line‑precise file references are encouraged in the developer instruction set (CLI renderer supports `path:line`). This aids both AI (grounded outputs) and human review (clickable, precise context in editors).
- Tests include a minimal C++ extension smoke check (`tests/test_cpp_extension.py`) — ensures the devcontainer toolchain is wired. Humans get early broken dev clues; agents can detect environment regressions quickly.
- “Do not revert files you didn’t edit” in `AGENTS.md` is agent‑specific hygiene and reduces merge churn. It also matches human PR etiquette.
- `Justfile` includes CPU‑only CI helper (`ci-cpu`) for torch with uv pip; this reduces friction in constrained environments. Agents can use it to triage without GPU; humans have a backstop for local runs.
- `docs/math/*` pages mirror module APIs in `src/viterbo/math/…`. This documentation shape helps humans reason about expected shapes/dtypes and lets agents cross‑reference during implementation.
- The repo clearly separates “briefs/” and “reviews/” (intent difference): briefs propose, reviews observe. This reduces confusion for humans and gives agents predictable document types to consult.
- `skills/performance-discipline.md` mandates measurement before C++ escalation and ties to benchmarks. Agents are less likely to prematurely optimize; humans have a shared escalation gate.
- Safety: `skills/basic-environment.md` and `AGENTS.md` stress “no `pip` in repo; prefer `uv`”, and streamed reads. This avoids environment snowflakes and excessive log output in UIs.
- Edge: No explicit “Always‑On Skills” are marked, and `AGENTS.md` renders the placeholder. This is a minor UX gap for agents that could be fixed by tagging `basic-environment` as always‑on.
- Edge: Some math modules are stubs or raise `NotImplementedError` (e.g., `src/viterbo/math/similarity.py:10`, `src/viterbo/math/polar.py:32`, `src/viterbo/math/volume.py:95`). For AI agents, these are clear extension points; for humans, this clarifies scope and avoids hidden failures.
- Edge: The reviews nav intentionally avoids listing each review in `mkdocs.yml`, relying on the index. Humans who prefer browsing via nav might miss new reviews unless the index is kept up to date — which the process enforces. Agents are unaffected.
- Edge: `skills` metadata validation warns but does not fail the build unless invoked via `--check` and interpreted; `mkdocs --strict` won’t catch skill metadata drift. Humans/agents should rely on `just lint` (which runs `--check`) to spot drift.
- Edge: The repo encourages `rg` with a 250‑line limit in shell outputs, which interacts with complex multi‑file reads; agents must plan reads accordingly (chunked), and humans should rely on editors.
- Edge: The `Justfile` omits a dedicated docs‑only fast gate (e.g., `just docs`), though `ci` runs docs. Agents often need `uv run mkdocs build --strict`; adding a small alias could improve ergonomics for both audiences.
- Edge: In `docs/workflows-project-owner-vibekanban.md`, some operational questions/answers are embedded as “Answers to current questions”; helpful for humans but a bit chatty for agents. Still acceptable due to clear sectioning.
- Edge: The Reviews Template is present, but some published reviews may not uniformly use it (topic 1 vs topic 8). Consistency helps both agents and humans to scan. The index calls out the template, but adoption depends on authors.
- Edge: No explicit “review rubric” or scoring; comparing review quality across topics remains qualitative. Humans may miss prioritization; agents will output long lists without severity tags.
- Edge: Consolidation of findings into actionable tickets is not automated. Agents could add “Needs‑Unblock” or “Owner” tags; humans need a manual pass.
- Edge: No explicit “AI vs Human” considerations page; this review fills the gap, but a short canonical doc could articulate the design stance (token budgets, deterministic commands, chunked outputs, click-to-open paths) to guide future contributors.
- Edge: The `nav` does not include `docs/README.md` explicitly; landing pages are navigable via “Home” (points to `README.md`). This is fine; agents rely on paths, humans can use search.
- Edge: Security posture is addressed in owner workflow docs but not centralized in a SECURITY.md. For agents/humans, the current notes suffice; a central policy might be useful later.
- Positive: `docs/architecture/overview.md` explains C++ fallback policy and shows a one‑liner to trigger the local build; this empowers humans to test toolchain and agents to validate at runtime.
- Positive: Task numbering and topic lists in `docs/reviews/README.md` set clear scaffolding; agents can autocomplete expected paths for new reviews; humans see the roadmap.
- Positive: The repo avoids over‑customization in MkDocs (Material defaults) and keeps docs simple. This usability is good for humans and helps agents avoid format‑specific quirks.
- Positive: The `skills` texts are concise, imperative, and clearly scoped. They are structured for agent consumption (short, action‑first) and remain readable for humans.
- Positive: The devcontainer includes VS Code Tunnel and Cloudflared flows documented end‑to‑end; humans can operate without local VS Code, and agents get a stable, remotely accessible environment.
- Positive: PyRight configs include a strict profile and a basic profile; humans/agents can choose the right depth (`pyrightconfig.strict.json`, `pyrightconfig.json`).
- Positive: `tests/performance/` includes smoke‑tier benchmarks; encourages evidence-based performance work (agents can record artifacts; humans can compare runs).
- Positive: `skills/environment-tooling.md` describes deviations and troubleshooting, useful when things drift. Good for both AI and humans facing environment hiccups.
- Positive: `skills/repo-layout.md` gives canonical paths and sources of truth; helps agents build accurate file references and humans to orient quickly.
- Positive: The repository clearly states “No Codespaces; no Codex Cloud” in `docs/environment.md` golden path — reduces ambiguity for human ops and agent expectations.

Actions (first pass)
- Mark `skills/basic-environment.md` as `relevance: always` to populate the “Always‑On Skills” section in `AGENTS.md`; optionally include `repo-onboarding`.
- Add a tiny `just docs` alias that runs `uv run mkdocs build --strict` for docs‑only validations.
- Consider a short `docs/policies/ai-vs-humans.md` stating design stance (deterministic commands, chunked outputs, file path references, evidence capture) and linking to `AGENTS.md` essentials.
- Optionally add a `docs/security.md` consolidating security notes (tokens, Cloudflared, bind mounts) currently spread across environment/workflow docs.
- Encourage consistent use of `docs/reviews/TEMPLATE.md` across all reviews; update older reviews to the template shape for consistent scanability.
- Tag one or two skills as “always‑on” to improve agent triage; keep the list small to avoid noise.
- Add a simple “Docs Validated” checkbox to the Reviews pre‑submit checklist, explicitly requiring `uv run mkdocs build --strict` output recorded in the review.

Cross-References
- AGENTS.md
- Justfile
- mkdocs.yml
- docs/reviews/README.md
- docs/workflows-project-owner-vibekanban.md
- docs/environment.md
- skills/basic-environment.md
- skills/vibekanban.md
- skills/good-code-loop.md
- skills/skill-authoring.md
- docs/architecture/overview.md
- src/viterbo/math/*, tests/* (for API/docs parity and markers)

Validation
- Loaded skills metadata: `uv run python scripts/load_skills_metadata.py --quiet` — OK
- Built docs strictly: `uv run mkdocs build --strict` — OK (post‑write)

Limitations
- Did not run `just checks` (docs‑only change); existing CI parity is unaffected.
- Did not deeply review math correctness; focus was on ergonomics, docs/process, and safety/formatting edges.
- Did not test Codespaces; review assumes the documented local devcontainer golden path.

Status
- Finalized
