Status: Draft
Title: Review — DevOps Justification Gaps (Flag ANY Unjustified Detail)
Author: Codex CLI Agent
Date: 2025-10-27

# Scope & Method
- Scope: end-to-end devops surfaces — Justfile, CI/CD, docs workflows, devcontainer, Cloudflare Workers, lint/type/test config, editor/tooling, secrets, and guardrails.
- Method: read configs/scripts under repo; cross-check against AGENTS.md, skills/, docs/milestones/current.md, docs/environment.md, and workflows; flag any detail that lacks a clear, proximate, readable justification. For each item: file:line, the unjustified detail, alternatives, and any partial justification found.

Note: Line numbers refer to the first relevant line only. “Partial justification” may reference reviews or comments but these are not considered authoritative for a new developer.

---

# High-Risk Mismatches or Breakages

- Justfile:405, 413 — Broken recipes
  - Unjustified/broken: `train-logreg`/`evaluate-logreg` reference `scripts/train_logreg_toy.py` and `scripts/evaluate_logreg_toy.py` which do not exist.
  - Alternatives: remove these targets; or add stub scripts with docs.
  - Partial: `.env.example` mentions WANDB but no “toy logreg” docs exist.

- .devcontainer/README.md:83 and .devcontainer/post-start.sh:108 — Non-existent script
  - Unjustified: References `owner-cloudflare-setup.sh` but the real helper is `container-admin cf-setup`.
  - Alternatives: fix README/script references; centralize Cloudflare guidance.
  - Partial: `container-admin cf-setup` exists and is the intended helper.

- Selector default mismatch — skills/docs vs Justfile
  - docs/milestones/current.md:26-30 and skills/basic-environment.md:14-21 suggest `just test` uses the incremental selector; Justfile:157-166 defaults to full smoke unless `JUST_USE_SELECTOR=1`.
  - Unjustified: Divergence not explained anywhere authoritative.
  - Alternatives: default `just test` to selector (with `test-full` for full smoke); or update skills/milestone to state opt-in.
  - Partial: skills/testing-and-ci.md mentions opt-in, but basic-environment contradicts it.

- Coverage enforcement vs Charter — inconsistent floors
  - CI `just ci-cpu` uses `--cov-fail-under=80` (Justfile:344) while Charter demands ≥95% for math and ≥60% for datasets (docs/charter.md:33-40).
  - Unjustified: No rationale for 80 nor for ignoring per-path floors.
  - Alternatives: enforce Charter floors (global and/or per-path) in CI and `just ci`.
  - Partial: reviews note the gap but are not policy.

- Base image suite mismatch for cloudflared install
  - container-admin adds Cloudflare APT repo for `focal` (Ubuntu 20.04) at .devcontainer/bin/container-admin:113; README also says focal (line 81), but base image is `devcontainers/universal:2` (likely jammy or newer).
  - Unjustified: No reason to pin to focal regardless of base.
  - Alternatives: detect suite (jammy, noble) or install via tarball with checksum.
  - Partial: README attempts to justify “deterministic install” but is out of sync.

---

# Detailed Findings (by file)

## Justfile

- Justfile:1-3 — Bash strict shell
  - Unjustified: `set shell := ["bash", "-euo", "pipefail", "-c"]` without rationale for requiring bash over sh/zsh.
  - Alternatives: default shell; POSIX script bodies; or document bash-only needs.
  - Partial: none.

- Justfile:12-19 — UV project env and link mode
  - Unjustified: Forces `UV_PROJECT_ENVIRONMENT` to `~/.cache/uv/project-envs/msc-math-viterbo` and `UV_LINK_MODE=hardlink` globally.
  - Alternatives: per-worktree `.venv` only; `UV_LINK_MODE=copy`; no override; document trade-offs.
  - Partial: workflows mention hardlink vs copy trade-off but not why global default is forced.

- Justfile:21 — Unused variable
  - Unjustified: `SMOKE_TEST_TIMEOUT` exported but not used in pytest flags.
  - Alternatives: wire `--timeout=${SMOKE_TEST_TIMEOUT}` or drop the var.
  - Partial: none.

- Justfile:26 — Pytest durations=10 in smoke
  - Unjustified: `--durations=10` without rationale.
  - Alternatives: enable only in CI; tune to goals; or omit.
  - Partial: none.

- Justfile:45-68 — VK worktree preflight policy
  - Unjustified: Enforces branch naming `vk/<worktree-name>` for worktrees; fails on detached HEAD/rebase; not documented in skills/docs.
  - Alternatives: warn-only; make policy documented; disable outside VK worktrees.
  - Partial: none.

- Justfile:118-126 vs 126-134 — Inconsistent formatting scope
  - Unjustified: `fix` formats `src tests scripts`; `format` and `lint` run on `.`.
  - Alternatives: standardize scope or document why asymmetry.
  - Partial: none.

- Justfile:146-148 — Redundant pyright args
  - Unjustified: Passes both `-p pyrightconfig.json` and `src/viterbo` even though config includes.
  - Alternatives: `pyright -p pyrightconfig.json` only; or rely on include.
  - Partial: none.

- Justfile:157-166 — Selector default (mismatch)
  - See High-Risk section.

- Justfile:168-173 — `test-no-cpp`
  - Unjustified: Special marker run not documented in skills/docs.
  - Alternatives: document use-cases; or drop.
  - Partial: none.

- Justfile:275-280 — Coverage target without floor
  - Unjustified: No `--cov-fail-under`; does not implement Charter floors.
  - Alternatives: add floors; per-path floors for math/datasets.
  - Partial: reviews note the gap.

- Justfile:290-301 — `just ci` parity
  - Unjustified: No coverage in local `ci`; CI enforces an 80 floor via `ci-cpu`.
  - Alternatives: align `just ci` with CI.
  - Partial: reviews say “mostly mirrored,” not sufficient.

- Justfile:307-358 — CPU-only torch via system site
  - Unjustified: Why use `uv pip install --system` with CPU index, not lock or `uv sync` in venv; pin to 2.5.1 while pyproject allows >=2.5.1,<2.6.
  - Alternatives: `uv sync --extra dev` with CPU indices; pin via lock.
  - Partial: comment “avoid CUDA downloads” but not lock divergence.

- Justfile:369-375 — docs-serve bind
  - Unjustified: `mkdocs serve -a 0.0.0.0:8000` for devcontainer; no reason to expose externally.
  - Alternatives: bind 127.0.0.1.
  - Partial: none.

- Justfile:381-390 — distclean scope
  - Unjustified: Destroys `.venv` and caches; no doc when safe to run.
  - Alternatives: document risks; scope to current worktree.
  - Partial: none.

- Justfile:424-447 — Cloudflare helpers
  - Unjustified: Duplicate entrypoints across Justfile and `.devcontainer/bin/*` without a single source of truth.
  - Alternatives: alias Just targets to bin scripts; document one canonical flow.
  - Partial: `.devcontainer/cloudflare/README.md` exists but has inaccuracies (see below).

- Justfile:251-259 — Release target
  - Unjustified: `uv version --bump`/git tag flow kept although package not published; no policy doc.
  - Alternatives: drop or mark experimental; document when to use.
  - Partial: build/publish targets are disabled, but release remains.

## GitHub Actions — CI

- .github/workflows/ci.yml:35-41 — Ensure bash step
  - Unjustified: Installing bash on ubuntu-latest; redundant.
  - Alternatives: drop; platform-conditional if needed.
  - Partial: none.

- .github/workflows/ci.yml:16-20 — Global indices
  - Unjustified: Exports PIP/UV CPU indices globally without docs; could be step-scoped.
  - Alternatives: per-step; rely on lock; explain CPU-only policy.
  - Partial: docs workflow comments mention avoiding CUDA downloads.

- .github/workflows/ci.yml:64-66, 344 — Coverage enforcement mismatch
  - See High-Risk section.

- .github/workflows/ci.yml:85-89 — Deep tests/benches on PR
  - Unjustified: Running deep tier and benches on every push/PR; risks CI SLO.
  - Alternatives: gate behind label/schedule; manual dispatch for deep.
  - Partial: Charter sets CI p95 ≤ 7 min; not reconciled here.

- .github/workflows/ci.yml:12-13 and 91-146 — Longhaul schedule
  - Unjustified: Weekly cron at “0 5 * * 1” without timezone or rationale.
  - Alternatives: document timezone; choose lower-traffic window.
  - Partial: none.

- .github/workflows/ci.yml:55-59 — Torch extensions cache key
  - Unjustified: Cache key omits compiler/Python version; no cache-bust policy.
  - Alternatives: include toolchain fingerprints; doc update policy.
  - Partial: none.

## GitHub Actions — Docs

- .github/workflows/docs.yml:43-55 — Mixed uv pip vs uv sync
  - Unjustified: Creates venv, then installs torch+dev deps with `uv pip` instead of `uv sync`; diverges from lock.
  - Alternatives: `uv sync --extra dev` with CPU indices; explain lock usage.
  - Partial: “avoid CUDA downloads” comment; not a lock policy.

- .github/workflows/docs.yml:57-68 — `VITERBO_CPU_LIMIT=0` usage
  - Unjustified: Good practice, but not referenced in central docs for notebook rendering.
  - Alternatives: link to milestone/tooling defaults; add a skills note.
  - Partial: milestone mentions CPU cap; not cross-linked.

## Devcontainer

- .devcontainer/devcontainer.json:3 — Base image
  - Unjustified: `mcr.microsoft.com/devcontainers/universal:2` (heavy) over Python-focused base; no trade-off doc.
  - Alternatives: python:3.12 devcontainer + Node feature; or custom image.
  - Partial: none.

- .devcontainer/devcontainer.json:4 — remoteUser
  - Unjustified: `codespace` user without Codespaces; no rationale vs default `vscode`.
  - Alternatives: use default or document why `codespace`.
  - Partial: none.

- .devcontainer/devcontainer.json:11-19 — Editor extensions
  - Unjustified: Required vs optional not specified (e.g., `openai.chatgpt`, `bloop.vibe-kanban`).
  - Alternatives: minimal set; docs on optional extensions.
  - Partial: none.

- .devcontainer/devcontainer.json:25-39 — Env overrides
  - Unjustified: Hard-coded XDG overrides, history, `UV_PROJECT_ENVIRONMENT`; no central rationale.
  - Alternatives: rely on defaults; document why overrides are necessary.
  - Partial: none.

- .devcontainer/devcontainer.json:41-55 — Bind mounts
  - Unjustified: Assumes host `/srv/...` structure; no portability guidance; brittle.
  - Alternatives: parametrize via env; document OS variants.
  - Partial: docs/environment.md describes `/srv/` convention, but not alternatives.

- .devcontainer/post-create.sh:13-22 — uv install via curl | sh
  - Unjustified: No version pin/checksum; supply-chain risk.
  - Alternatives: pin version, checksum, or vendor in base image.
  - Partial: none.

- .devcontainer/post-create.sh:29-40 — Misc packages
  - Unjustified: Installs tmux, ripgrep, poppler-utils whenever apt present; not documented as required.
  - Alternatives: feature flags or docs; install on demand.
  - Partial: none.

- .devcontainer/post-create.sh:42-56 — `@openai/codex` global install
  - Unjustified: Not required by repo; may drift.
  - Alternatives: omit or document optional usage.
  - Partial: none.

- .devcontainer/post-start.sh:108 — Wrong helper name
  - See High-Risk section.

- .devcontainer/bin/container-admin:95-119 — Installers
  - Unjustified: VS Code CLI and `just` installed via curl without pin/checksum; Cloudflared via focal repo regardless of base.
  - Alternatives: pin versions; suite-detect; or pre-bake into base image.
  - Partial: README claims determinism but mismatched suite.

- .devcontainer/bin/container-admin:168-220 — Start policy
  - Unjustified: Auto-starts VS Code tunnel and cloudflared unless binaries missing; no security policy on unattended tunnels.
  - Alternatives: opt-in flags (default off); document risk posture.
  - Partial: README shows workflows but not policy.

- .devcontainer/bin/container-admin:284-313 — Status heuristics
  - Unjustified: Curl timeout `-m 1` and port checks are ad-hoc; no healthy criteria doc.
  - Alternatives: define health checks and expected outputs.
  - Partial: none.

- .devcontainer/bin/host-admin:1-200 — Host dependencies and `/srv`
  - Unjustified: Requires npm for devcontainer CLI install and assumes `/srv/devhome` roots on host.
  - Alternatives: OS-specific docs; env overrides; add Windows/macOS guidance.
  - Partial: README mentions host-setup but not portability.

- .devcontainer/README.md:81, 83 — Focal suite and wrong script
  - See High-Risk section.

## Cloudflare Workers

- .devcontainer/cloudflare/wrangler.toml:1-8 — Hard-coded routes
  - Unjustified: Directly references `vibekanban.joernstoehler.com/*`; no env separation.
  - Alternatives: wrangler environments or env vars; staging/prod routes.
  - Partial: README says “Replace with your hostname,” but not how.

- .devcontainer/cloudflare/wrangler-sanitizer.toml:1-7 — Same
  - Unjustified: API routes hard-coded; no env separation.
  - Alternatives: as above.
  - Partial: minimal comment only.

- .devcontainer/cloudflare/worker-vk-sanitizer.js:1-160 — Field allowlist and policy
  - Unjustified: Target keys `{title, description, comment, body, text}` hard-coded; no source-of-truth mapping to VK API schema.
  - Alternatives: schema-driven; KV-configured; document versioning policy.
  - Partial: README lists policy/intended behavior.

- .devcontainer/cloudflare/worker-font-injector.js:1-44 — Google Fonts injection
  - Unjustified: Pulls from Google Fonts; no CSP/privacy considerations; not configurable.
  - Alternatives: self-host fonts; toggle via env; document trade-offs.
  - Partial: none.

- .devcontainer/cloudflare/README.md:32, 49-50 — Wrong script names
  - Unjustified: References `.devcontainer/bin/admin` which doesn’t exist.
  - Alternatives: point to host-admin/container-admin; unify docs.
  - Partial: none.

## Packaging, Lint, Type, Coverage

- pyproject.toml:12 — Python >=3.12
  - Unjustified: No rationale for excluding 3.11 (PyTorch supports 3.11).
  - Alternatives: `>=3.11`; or document 3.12-specific features.
  - Partial: none.

- pyproject.toml:14-19 — Dependency pins
  - Unjustified: Torch range `>=2.5.1,<2.6` while CI pins 2.5.1; other deps have floor pins without lock-only policy doc.
  - Alternatives: rely on lock; or document floor rationale.
  - Partial: lock exists; not referenced as policy.

- pyproject.toml:41-44 — Namespace packages
  - Unjustified: `namespaces=true` without policy doc.
  - Alternatives: standard packages; doc why namespaces.
  - Partial: none.

- pyproject.toml:45-74, 88-96 — Ruff rules and ignores
  - Unjustified: Rule set and per-file ignores lack policy doc (why `ANN401`, `TID252`, width=100, etc.).
  - Alternatives: link to a lint policy; briefly justify in skills/good-code-loop.md.
  - Partial: reviews say “correctness-first,” not canonical.

- pyproject.toml:112-114 — Coverage config minimal
  - Unjustified: No `[tool.coverage.report]` or floors; doesn’t reflect Charter.
  - Alternatives: add report config and floors; per-path if needed.
  - Partial: Charter defines floors.

- pyrightconfig.json:4-5,9-10 — Exclude and extraPaths
  - Unjustified: Excludes `src/viterbo/exp1`; no rationale. `extraPaths` includes tests — good — but not documented.
  - Alternatives: document ignore policy; prune dead excludes.
  - Partial: none.

## Pytest

- pytest.ini:3 — addopts
  - Unjustified: `-q -ra --maxfail=1 --import-mode=importlib`; no policy doc.
  - Alternatives: document test culture reasons.
  - Partial: none.

- pytest.ini:5 — timeout=120
  - Unjustified: 120s global timeout without rationale, esp. given sitecustomize CPU cap.
  - Alternatives: tie to SLAs; make configurable via env (SMOKE_TEST_TIMEOUT already defined but unused).
  - Partial: none.

- pytest.ini:12-13 — DeprecationWarning as error
  - Unjustified: Policy not documented relative to dependency churn impact.
  - Alternatives: narrow scope; document rationale.
  - Partial: reviews mention this posture, not policy.

## Pre-Commit

- .pre-commit-config.yaml:2-19 — Hook versions
  - Unjustified: No bump cadence/policy; pinned to specific tag versions.
  - Alternatives: document update policy; enable dependabot.
  - Partial: none.

- .pre-commit-config.yaml:23-27 — pytest on pre-push
  - Unjustified: `just test` default is full smoke; can slow pushes; mismatch with “incremental by default” docs.
  - Alternatives: use `just test-fast`; or default selector on.
  - Partial: none.

- .pre-commit-config.yaml:28-32 — Secret blocking scope
  - Unjustified: Only blocks WANDB_API_KEY; no broader secret scanning policy.
  - Alternatives: add detect-secrets or trufflehog; doc policy.
  - Partial: `.env` is gitignored; still risky in tools.

## Editor/Repo Hygiene

- .vscode/settings.json:17,33 — `artefacts` visible
  - Unjustified: Explorer/search `artefacts` not excluded while other generated dirs are; no doc why visible by default.
  - Alternatives: exclude by default; advise how to toggle.
  - Partial: none.

- .gitattributes:1-23 — LFS patterns
  - Unjustified: No LFS policy doc (retention, contribution rules).
  - Alternatives: add short LFS policy under docs/.
  - Partial: skills/experiments-notebooks-artefacts.md covers storage norms, not LFS specifics.

- .editorconfig:8,13-15 — Indentation policy
  - Unjustified: 4-space default, 2-space YAML/JSON; no coding standards doc.
  - Alternatives: document in skills/good-code-loop.md.
  - Partial: none.

## Guardrails & Utilities

- sitecustomize.py:39 — CPU cap 180s
  - Unjustified: Why 180s specifically; not tied to CI SLO or notebooks.
  - Alternatives: document rationale; tune per-tier; or disable by default for non-test code.
  - Partial: milestone mentions cap and env var override.

- sitecustomize.py:62-70 — JAX enable_x64
  - Unjustified: Configures JAX though repo is Torch-first; no doc that JAX is used.
  - Alternatives: only if JAX in dependencies; move to JAX-specific flows.
  - Partial: comment says “avoid dtype bugs,” not scoped to this project.

- scripts/cpp_preflight.py:38-47 — Build flags
  - Unjustified: Forces `-O0` and `USE_NINJA=1`; no doc on why these flags are chosen for preflight.
  - Alternatives: document choice or use defaults; add toolchain fingerprints to cache.
  - Partial: none.

## Notebooks Rendering

- scripts/render_notebooks.py:63-71 — `--hide-input` and `|| true` in CI
  - Unjustified: CI uses `--hide-input || true` (docs workflow) suppressing errors during render.
  - Alternatives: fail-fast for notebooks; or justify tolerance.
  - Partial: skills note CI renders notebooks; no error policy documented.

## Documentation & Workflows

- docs/milestones/current.md:26-30 — Selector default claim
  - See mismatch in High-Risk section.

- docs/environment.md:15, 31 — Data dir path mismatch
  - Unjustified: Uses `~/.local/share/ai/bloop/vibe-kanban` while devcontainer mounts `~/.local/share/vibe-kanban` (no `ai/bloop`).
  - Alternatives: converge on one path; document migration if needed.
  - Partial: none.

- docs/workflows-project-owner-vibekanban.md:57-68 — Cloudflare quick vs named tunnel
  - Unjustified: Shows both ephemeral quick tunnel and named tunnel without policy on which to prefer.
  - Alternatives: state policy (prefer named; quick only for experiments).
  - Partial: none.

- scripts/README.md — Migration note
  - Unjustified: Says “During migration, DevOps scripts are intentionally minimal” but no defined end-state or acceptance criteria.
  - Alternatives: capture a target state checklist.
  - Partial: none.

## Secrets & Env

- .env.example:3 — Stale Makefile reference
  - Unjustified: Mentions “Make targets” though this repo uses Justfile.
  - Alternatives: update to “Just targets” or remove.
  - Partial: none.

- .env:6 — Real-looking WANDB_API_KEY present (redacted here)
  - Risk: Although .env is in .gitignore, presence is dangerous and not documented as acceptable.
  - Alternatives: keep .env out of repo; use secret managers; rely on `.env.example` only.
  - Partial: pre-commit hook blocks staging of WANDB key, but scope is narrow.

---

# Suggested Next Actions (non-binding)
- Align selector default across docs and Justfile. Either make selector default-on in `just test` or correct docs/milestone to explicitly call `just test-fast`.
- Remove or stub broken logreg recipes.
- Add coverage floors per Charter; mirror in `just ci` and CI workflow. Consider per-path floors for math/datasets.
- Fix Cloudflare docs/scripts: replace references to `owner-cloudflare-setup.sh` with `container-admin cf-setup`; correct `.devcontainer/cloudflare/README.md` script names; parameterize routes; align apt suite to base image.
- Document and/or relax hard-coded environment overrides: UV project env path, XDG overrides, bind mounts, remoteUser.
- Replace curl | sh installers with pinned versions and checksums; or move to base image build.
- Clarify Python version policy (3.12 vs 3.11) and Torch pin strategy (lock vs floors).
- Add a short “Lint/Type policy” doc (why chosen Ruff rules, per-file ignores, width=100).
- Decide on notebook render failure policy; remove `|| true` if we want strict docs.
- Resolve data-dir path mismatch (`ai/bloop` segment) across environment + devcontainer.

