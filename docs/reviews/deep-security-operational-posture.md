# Review — Security & Operational Posture Deep‑Dive

Provenance
- Topic: Security & Operational Posture Deep‑Dive
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 87fdebc21a0c91998e746078008fa7067760b1d3
- Timebox: ~60 minutes
- Scope: secrets handling (.env, tokens), cloudflared/tunnel workflows, VK sanitizer, supply chain (uv/torch wheels), CI permissions, documented safety/formatting practices
- Version: v0.1

Context
- Deep‑dive review requested to surface concrete, actionable security and operations issues across secrets, tunnels, edge sanitizer, supply chain pinning, CI perms, and safety/formatting docs.

Inputs & Method
- Read AGENTS and skills index to align on workflow and expectations: `AGENTS.md`, `skills/*.md`.
- Surfaced skills metadata: `uv run python scripts/load_skills_metadata.py --quiet --print`; checked sections: `uv run python scripts/load_skills_metadata.py --check --quiet`.
- Searched codebase for cloudflared/tunnel/worker usage: `rg -n "cloudflared|tunnel|cloudflare" -S`.
- Inspected Cloudflare Workers and wrangler configs: `.devcontainer/cloudflare/*`.
- Reviewed devcontainer lifecycle scripts: `.devcontainer/bin/container-admin`, `.devcontainer/post-*.sh`, `.devcontainer/devcontainer.json`.
- Checked CI and Pages workflows, cache, and index settings: `.github/workflows/*.yml`.
- Scanned secrets patterns and env handling: `.env`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`.
- Validated docs build: `uv run mkdocs build --strict` — OK.

Findings (unsorted)
- Critical: A real‑looking secret is committed in `.env:6` (`WANDB_API_KEY="025a..."`). `.gitignore` ignores `.env`, but ignore rules do not apply to already‑tracked files. Immediate rotation + history purge required; treat key as compromised.
- `.env.example` exists and is correct as a template (`WANDB_API_KEY="your-wandb-api-key"`), but `Justfile:train-logreg` encourages loading `.env`. Ensure `.env` is never tracked again (add pre‑commit secret scan, server‑side protection).
- No secret scanning in pre‑commit; `.pre-commit-config.yaml` lacks `detect-secrets`/`gitleaks` hooks. Also no GH Action for secret scanning. This gap likely enabled the `.env` leak.
- CI permissions: `docs.yml` explicitly sets `permissions: { contents: read, pages: write, id-token: write }` which is least‑privilege for Pages. `ci.yml` does not declare permissions; defaults to `contents: read` on newer repos, but making it explicit is preferable for defense‑in‑depth.
- Cloudflare Access is documented (Zero Trust app gating `https://vibekanban.joernstoehler.com`) in `docs/environment.md:45` and `docs/workflows-project-owner-vibekanban.md:Cloudflare Access` — strong control if actually configured. There’s no automated/assertion check in scripts to verify Access is in place; relies on operator discipline.
- Cloudflared workflows are concrete and state is persisted to host mounts (`/srv/devhome/.cloudflared` and `.config/.wrangler`). `post-start.sh` chowns mounts idempotently and refuses to run on host — good hygiene.
- Named tunnel config helper (`.devcontainer/bin/container-admin:cmd_cf_setup`) writes `config-<tunnel>.yml` and wires DNS via `cloudflared tunnel route dns`. This reduces manual steps; however, the helper assumes tunnel already exists and that login has been performed (clear error message is printed when missing).
- Package installation in devcontainer relies on network bootstrap scripts:
  - `uv` install via `curl | sh` (astral.sh) in `post-create.sh` is common but carries supply‑chain risk. Consider pinning a specific version and verifying signature/checksum.
  - `just` installed via `curl | bash` in `container-admin` (fallback) similarly lacks checksum verification.
  - `wrangler` installed globally via `npm i -g wrangler` without version pinning; API changes could break deployments. Pin `wrangler@<version>`.
- Cloudflared installation uses Cloudflare APT repo with `signed-by` key (good), but hardcodes Ubuntu suite `focal` (`.devcontainer/bin/container-admin:ensure_cloudflared`, `.devcontainer/README.md`). If the base image changes (e.g., jammy), the suite should match or installation may drift. Consider parametrizing suite or switching to Cloudflare’s recommended install for the image in use.
- VibeKanban UI bind address defaults to `0.0.0.0` (`.devcontainer/devcontainer.json`, `container-admin`). Exposure is expected because Cloudflare proxies bind locally; ensure host firewall blocks direct exposure on the host if the container is reachable outside Zero Trust (Tailscale/SSH guidance covers this, but the risk exists if host policy is lax).
- Cloudflare Worker: font/CSS injector (`.devcontainer/cloudflare/worker-font-injector.js`) pulls Google Fonts from `fonts.googleapis.com` and `fonts.gstatic.com`. External calls introduce privacy/CDN dependency; consider self‑hosting the font (KV or Pages Asset) or inlining minimal CSS to avoid third‑party beacons.
- Cloudflare Worker: VK sanitizer (`.devcontainer/cloudflare/worker-vk-sanitizer.js`) scopes to `POST|PUT|PATCH` under `/api/*`, escapes intraword underscores, and preserves code spans, fenced code blocks, link destinations, angle autolinks, and bare URLs. This is thoughtfully constrained.
- Sanitizer edge cases to watch:
  - Reference‑style links (`[text][id]`) are not explicitly handled; likely fine if VK doesn’t support reference IDs.
  - Destinations containing nested parentheses may break the simple `(...)` capture; uncommon in VK content, but possible.
  - Email addresses like `a_b@example.com` in plain text could receive escapes (not in angle brackets), altering displayed text. If this appears in practice, extend preserved ranges to email patterns.
  - JSON fields beyond the allowlist (`title`, `description`, `comment`, `body`, `text`) will not be sanitized; that’s correct by default but requires vigilance if VK adds new Markdown fields (docs call this out).
- Wrangler routes are specific and minimal: `wrangler-sanitizer.toml` applies to `/api/*`, `wrangler.toml` applies to `/*` for the same host. Good separation of concerns; collisions unlikely.
- Optional auto‑deploy of workers on service start via `CF_AUTO_DEPLOY=1` (`container-admin:cmd_start`) is convenient but risky (implicit deploys). Keep default off; if ever enabling, prefer a dedicated manual `just`/admin command to avoid accidental redeploys.
- Devcontainer mounts show a data path inconsistency:
  - `devcontainer.json:mounts` uses `~/.local/share/vibe-kanban` (without `/ai/bloop/`).
  - `post-start.sh` fixes ownership for `~/.local/share/ai/bloop/vibe-kanban` but checks mount presence for `~/.local/share/vibe-kanban` later. Inconsistent paths risk data non‑persistence or permission drift; pick one canonical path and align docs/scripts/mounts.
- CI supply‑chain posture:
  - `ci.yml` preconfigures `UV_DEFAULT_INDEX`/`PIP_INDEX_URL` to PyTorch CPU wheels for deterministic installs; then uses `ci-cpu` to install `torch==2.5.1` system‑wide and `-e .[dev]` for the rest. Torch is version‑pinned (good), but installs are not hash‑pinned.
  - `uv.lock` exists and pins hashes for packages (including torch wheel URLs). However, `ci-cpu` bypasses the lock for torch; keep torch pin aligned with `uv.lock` and consider verifying hashes when feasible.
  - Actions cache (`~/.cache/uv`) keyed by `uv.lock` (good); cache poisoning risk is minimal when scoped to runner and action.
- Docs quality controls are strong: `mkdocs.yml` strict build is required in CI and mirrored locally (`Justfile:ci`, `docs.yml`). Local build here: `uv run mkdocs build --strict` — OK; warnings only about orphan pages not in nav (policy‑consistent).
- Formatting/type/testing practices are well‑documented and automated:
  - `just fix|format|lint|type|test` scripts exist and mirror CI; `precommit-slow` is a sensible local gate.
  - Ruff config focuses on correctness and policy checks; pyright basic is used by default with a strict variant available.
  - Skills and AGENTS emphasize running `just checks` and `just ci` pre‑handoff.
- Devcontainer image `mcr.microsoft.com/devcontainers/universal:2` is not pinned to a digest; base image drift can alter tooling and APT behavior over time. Consider pinning to a digest or a dated tag to stabilize supply.
- No SLSA/provenance or artifact signing configured for built documentation or Python distributions (build disabled); acceptable for research, but note for future hardening.
- `.github/workflows/docs.yml` uses `actions/deploy-pages@v4` with `id-token: write`; standard and appropriate.
- No automated secret detection in CI; adding a weekly `gitleaks` or `trufflehog` scan would complement pre‑commit and catch regressions.
- The repo already flagged the `.env` leak in another review (`docs/reviews/14-misc.md`), but it remains present; action did not occur yet. Treat remediation as urgent.

Actions (first pass)
- Revoke and rotate the leaked `WANDB_API_KEY`; remove `.env` from history (e.g., `git filter-repo`), replace with placeholders only; keep `.env.example` as the template.
- Add secret scanning: pre‑commit hook (`detect-secrets` or `gitleaks`) and a CI job running weekly and on PRs.
- Normalize VibeKanban data dir path across `devcontainer.json`, `post-start.sh`, and docs; choose either `~/.local/share/vibe-kanban` or `~/.local/share/ai/bloop/vibe-kanban` and update mounts/ownership logic accordingly.
- Pin and verify installer tooling: use `wrangler@<version>`, pin `just` installer release with checksum, and pin `uv` installer version or verify its signature. Parameterize Cloudflared APT suite to match the base image.
- Make CI permissions explicit in `.github/workflows/ci.yml` (`permissions: { contents: read }`) and audit job steps for least privilege.
- Consider serving Inter font without third‑party calls (Cloudflare KV/Pages Asset or inline CSS) to avoid `fonts.googleapis.com`/`gstatic` dependencies.
- Keep `CF_AUTO_DEPLOY` default off; if desirable, gate behind an explicit `just cf` or admin subcommand only (no auto on start).

Cross‑References
- Files: `.env:6`, `.env.example:6`, `.gitignore:6`, `.pre-commit-config.yaml:1`, `.github/workflows/ci.yml:1`, `.github/workflows/docs.yml:9`, `.devcontainer/devcontainer.json:1`, `.devcontainer/post-start.sh:1`, `.devcontainer/bin/container-admin:1`, `.devcontainer/cloudflare/worker-vk-sanitizer.js:1`, `.devcontainer/cloudflare/worker-font-injector.js:1`, `.devcontainer/cloudflare/wrangler-sanitizer.toml:1`, `.devcontainer/cloudflare/wrangler.toml:1`, `Justfile:212`, `mkdocs.yml:1`, `AGENTS.md:1`.
- Topics: skills/testing-and-ci.md, skills/basic-environment.md, skills/environment-tooling.md, skills/vibekanban.md, docs/workflows-project-owner-vibekanban.md.

Validation
- `uv run python scripts/load_skills_metadata.py --quiet --print` — OK (no output by design)
- `uv run python scripts/load_skills_metadata.py --check --quiet` — WARN: always‑on list empty (expected)
- `uv run mkdocs build --strict` — OK (site builds; orphan pages noted as expected)
- `rg -n "cloudflared|tunnel|cloudflare" -S` — surfaced worker/tunnel docs and commands (see files above)

Limitations
- Did not run `just checks`/test suites to avoid lengthy runs; focus stayed on security posture and ops scripts/docs.
- Did not verify Cloudflare Access policy live; treated docs as source of truth.
- Did not exercise the sanitizer against live VK API; reasoning based on code inspection.

Status
- Draft

Pre‑Submit Checklist
- Link in `docs/reviews/README.md` — Pending (guardrails: no index edits in this task)
- `uv run mkdocs build --strict` — OK
- `just checks` — N/A (docs‑only change)
- Actions extracted — Yes (above)
- Cross‑refs noted — Yes
- Provenance filled — Yes
- Title matches pattern — Yes

