# Project Owner Workflow: VibeKanban + Devcontainer + Remote Access

This document defines the end‑to‑end “Project Owner” workflow using VibeKanban as the canonical backlog, with agents executing work via Codex CLI inside the devcontainer. It is intended to be unambiguous, correct, and ready to follow as a daily runbook.

Scope
- VibeKanban runs inside the devcontainer and exposes a web UI.
- Codex agents are launched by VibeKanban and operate in per‑task git worktrees under `/var/tmp/`.
- Access from anywhere uses Tailscale + SSH, VS Code Tunnel, and Cloudflare for public HTTPS to VibeKanban.
- Tokens and heavy caches persist across container rebuilds via devcontainer volumes.

Components
- Dev host: Ubuntu workstation with Tailscale daemon running.
- Devcontainer: the project’s container (Python 3.12, uv, etc.), started via `devcontainer` CLI.
- VibeKanban: official BloopAI app, distributed as a prebuilt Rust+React binary via `npx vibe-kanban`; also ships an MCP server (`npx vibe-kanban --mcp`).
- Codex CLI agent: runs inside the devcontainer, with MCP access to VibeKanban.
- VS Code Tunnel: remote editor session to the devcontainer.
- Cloudflared: public HTTPS for VibeKanban at `vibekanban.joernstoehler.com`.

Prerequisites (one‑time on the host)
- Install Tailscale and log in (`tailscale up`); verify SSH from remote.
- Install `devcontainer` CLI (VS Code Dev Containers CLI).
- Prepare host directories for persistent state (mirror container HOME):
  - `/srv/devhome/.config/gh` for `gh` CLI auth
  - `/srv/devhome/.vscode` for VS Code Tunnel state
  - `/srv/devhome/.config/codex` for Codex CLI auth
  - `/srv/devhome/.config/.wrangler` for Cloudflare Wrangler config/state
  - `/srv/devhome/.cloudflared` for Cloudflare tunnel credentials
  - `/srv/devhome/.cache/uv` for `uv` cache (PyTorch wheels, etc.)
  - `/srv/devhome/.local/share/ai/bloop/vibe-kanban` for VibeKanban data
  - Worktrees base: `/srv/devworktrees/vibe-kanban/worktrees`
  - `sudo chown -R $USER:$USER /srv/devhome /srv/devworktrees` after first creation

Start the environment
- Recommended (one-shot from host):
  - `bash .devcontainer/bin/host-admin up preflight start --interactive`
  - Brings up the devcontainer, runs preflight, starts VS Code Tunnel + Cloudflared + VibeKanban (detached), then verifies.
- Low-level alternative:
  - On host: `devcontainer up --workspace-folder /srv/workspaces/msc-math-viterbo`
  - Enter shell: `devcontainer exec --workspace-folder /srv/workspaces/msc-math-viterbo bash -l`

Inside the devcontainer: baseline bootstrap
- Ensure tooling (post‑create already handles most of this):
  - `just sync` (idempotent uv sync)
  - `just checks` (format, lint, type, smoke tests)
- Optional: pin PyTorch index if needed (CPU baseline):
  - `export UV_DEFAULT_INDEX=https://download.pytorch.org/whl/cpu`
  - `export UV_EXTRA_INDEX_URL=https://pypi.org/simple`

Install and run VibeKanban (official, via npx)
- No clone required to run: the app is shipped via npm.
- Start the app inside the devcontainer (choose a port and bind address):
  - `HOST=0.0.0.0 FRONTEND_PORT=3000 npx vibe-kanban`
  - The backend port is auto‑assigned by default; the frontend serves on `FRONTEND_PORT` (default 3000).
  - Access inside the container: `http://127.0.0.1:3000` (or your chosen port).
- MCP server only (for editor integrations), if needed separately:
  - `npx vibe-kanban --mcp`
  - Use when you want the MCP tools without the full UI process.

Expose VibeKanban via Cloudflare
- One‑time: authenticate Cloudflared inside the devcontainer (persisted via volume):
  - `cloudflared tunnel login`
- Ephemeral quick tunnel for dev:
  - `cloudflared tunnel --url http://127.0.0.1:3000 --hostname vibekanban.joernstoehler.com`
- For a named tunnel (recommended long‑lived):
  - `cloudflared tunnel create vibekanban`
  - `cloudflared tunnel route dns vibekanban vibekanban.joernstoehler.com`
  - `cloudflared tunnel run vibekanban`
  - The credentials JSON is stored under `~/.cloudflared/` (persisted).

Persist VibeKanban data on the host (no Docker volumes)
- Bind‑mounts are preconfigured in `.devcontainer/devcontainer.json`:
  - App data directory (Linux): `~/.local/share/ai/bloop/vibe-kanban/` ← `/srv/devhome/.local/share/ai/bloop/vibe-kanban`
  - Worktrees: `/var/tmp/vibe-kanban/worktrees` ← `/srv/devworktrees/vibe-kanban/worktrees`
  - These keep state across container rebuilds and make backups easy.

VS Code Tunnel
- Start inside the devcontainer:
  - `code tunnel --accept-server-license-terms --name viterbo-dev`
- Connect from any browser to the Code Tunnel URL.
- Opening a worktree in the remote session:
  - Either use the web UI link (if supported), or run inside the container:
    - `code --add /var/tmp/vibe-kanban/worktrees/<task-slug-or-id>`

Agent flow with VibeKanban
- You, the Owner, work in the browser at `vibekanban.joernstoehler.com`.
- Create/select a project and tickets in the Kanban board.
- Click “Start” on a ticket:
  - VibeKanban launches the Codex agent inside the devcontainer.
  - A per‑task git worktree is created under `/var/tmp/vibe-kanban/worktrees/<slug-or-id>`.
  - The agent runs with full repo context, authenticated tooling, uv, etc.
  - When done, the UI shows diffs, offers to merge, or to open the worktree in VS Code.
- MCP usage from agents (illustrative):
  - The MCP server (`npx vibe-kanban --mcp`) exposes tools to create/update/list tickets and add comments.
  - Tickets can reference long briefs stored in `docs/briefs/` via links, keeping tasks short.

Backlog structure changes
- Canonical backlog moves to VibeKanban (projects + tickets).
- Long specifications, surveys, and algorithm docs live in `docs/briefs/`.
- Tickets reference briefs by path (e.g., `docs/briefs/2025-10-12-workflow-capacity-algorithms.md`).
- The old `docs/tasks/` tree has been retired; any straggling briefs should be migrated into tickets before merging.

Performance and caches (uv, PyTorch)
- Goal: fast cold starts while avoiding cross‑filesystem hardlink warnings.
- You currently cache uv under a host bind (`/srv/devhome/.cache/uv -> ~/.cache/uv`) and keep `.venv` under the repo (workspace FS). This leads to:
  - Warning: “cannot hardlink across filesystems”; uv falls back to copying. This adds ~5s on first sync per session.
- Acceptable options:
  - Do nothing: tolerate the small penalty; document it as expected.
  - Co‑locate `.venv` and the uv cache on the same mounted path as seen by uv. Two simple ways:
    - Keep both inside the workspace bind mount: set `UV_CACHE_DIR=.uv-cache` (add `.uv-cache` to `.gitignore`) and keep `.venv` in the repo. This guarantees same mount and enables hardlinks; persists via the host repo folder.
    - Or keep both under `$HOME` and bind‑mount `$HOME` (or a subfolder) from a host path, e.g., `/srv/devhome` mirrored to `/home/codespace`. Then `.venv` (if moved to `$HOME/venvs/project`) and `~/.cache/uv` share the same mount.
  - Using “host folder” vs “Docker volume” alone does not guarantee hardlinks; co‑location on the same mount point does.
- Recommendation: document that the warning is OK and expected with current mounts; revisit co‑location if first‑run cost grows materially.

Daily start sequence (owner)
- Fast path: `bash .devcontainer/bin/host-admin up preflight start --interactive` (host)
- Or inside the container:
  - Preflight: `bash .devcontainer/bin/container-admin preflight`
  - Start all (detached): `bash .devcontainer/bin/container-admin start --detached`
  - Verify: `bash .devcontainer/bin/container-admin status`
  - Then open `https://vibekanban.joernstoehler.com` and work the board.

Migration from docs/tasks to VibeKanban (completed 2025-10-16)
- All legacy briefs were converted to tickets in the `Msc Math Viterbo` project and the `docs/tasks/` directory was removed.
- New scoping notes belong in `docs/briefs/` (or directly on tickets) with links from the board.
- If a feature branch still introduces `docs/tasks/*`, stop and migrate that content into VibeKanban before landing.

Font choice in the VibeKanban UI
- KISS, no fork: inject Inter at the edge using Cloudflare Wrangler. We provide a worker and config under `.devcontainer/cloudflare/`:
  - Deploy in container: `cd .devcontainer/cloudflare && wrangler deploy`
  - Tail logs: `cd .devcontainer/cloudflare && wrangler tail`
  - Edit `wrangler.toml` to adjust the route/domain if needed.

VS Code “Open in Remote” button (from VibeKanban)
- The server could spawn `code --add /var/tmp/vibe-kanban/worktrees/<id>` inside the container, which should attach to the active Code Tunnel session if one is connected.
- However, web → VS Code handoff from the VibeKanban UI is not essential; the reliable path is:
  - Use the visible worktree path and run `code --add <path>` in the devcontainer shell.
  - Or copy‑paste the path into the VS Code remote file picker.
- Treat “Open in VS Code” as a nice‑to‑have feature; don’t block on it.

Troubleshooting
- VibeKanban not reachable publicly:
  - Check `cloudflared tunnel list`; ensure the tunnel is running and DNS is set for `vibekanban.joernstoehler.com`.
  - Verify VibeKanban is listening on `0.0.0.0:3000` (or your chosen `FRONTEND_PORT`).
- MCP tools not visible to agents:
  - Run `npx vibe-kanban --mcp` for the standalone MCP server if your editor expects stdio MCP.
  - Confirm the data directory exists at `~/.local/share/ai/bloop/vibe-kanban/` (Linux) and contains `db.sqlite`.
- uv hardlink warnings:
  - Expected if `.venv` and `~/.cache/uv` are on different filesystems. Either ignore or co‑locate directories.

Security notes
- Keep `~/.cloudflared/` on a private volume with limited access.
- Persist `gh`, VS Code Tunnel, and Codex CLI tokens in separate volumes; do not commit or bake into images.
- VibeKanban data persists under the OS app data dir (Linux: `~/.local/share/ai/bloop/vibe-kanban/`). Bind‑mount this path to a host folder for durability and backups.

Answers to current questions
- Better font in VibeKanban: yes, without forking. Use a Cloudflare Worker on `vibekanban.joernstoehler.com/*` to inject Inter CSS at the edge (no changes to upstream binaries). Deploy both Workers together via `bash .devcontainer/bin/host-admin cf` (host) or inside container via `bash .devcontainer/bin/container-admin cf`.
- Outbound text sanitizer: `.devcontainer/cloudflare/worker-vk-sanitizer.js` proxies `POST/PUT/PATCH` JSON under `/api/*` and escapes only intraword underscores. It preserves inline code, fenced code, and leaves underscores inside Markdown link destinations and URLs untouched. See `.devcontainer/cloudflare/README.md` for details. Optional auto-deploy: set `CF_AUTO_DEPLOY=1` and run `bash .devcontainer/bin/host-admin start` (host) or `bash .devcontainer/bin/container-admin start` (container).

See `.devcontainer/cloudflare/README.md` for the full rationale, decisions, and operational notes (routes, preserved contexts, allowlist, troubleshooting, and update strategy).
- “Open in VS Code” via GUI: non‑essential. CLI `code --add <worktree>` works reliably with Code Tunnel. Building a button that shells out to `code --add` in the server process is feasible but optional.
- uv sync hardlink warning: expected with current mounts. Either accept the ~5s penalty or co‑locate `.venv` and `~/.cache/uv` to enable hardlinks. Document as OK in this workflow.
- Documenting the workflow: this file is the canonical workflow; link it when onboarding agents and in ticket “why”.
- MCP quality: upstream `--mcp` binary exposes tools for list/create/update/comment. For Codex CLI, the `vibe_kanban` tools support create/list/update and starting task attempts. If we see gaps (bulk migration, richer fields), we can add MCP helper scripts.
- Task migration: complete. Continue creating/updating tickets directly in VibeKanban and keep supporting material in `docs/briefs/`.
- Faster daily bring‑up: the current stack is solid. The fastest reliable loop remains Tailscale + SSH → devcontainer CLI → start independent components (VibeKanban, Code Tunnel, Cloudflared). Use `bash .devcontainer/bin/host-admin up preflight start --interactive` (host) or `bash .devcontainer/bin/container-admin start --detached` (inside) to start all in one go.

 Suggested improvements
- Compose bring‑up/tear‑down via the simplified `.devcontainer/bin` scripts:
  - Start all (detached): `bash .devcontainer/bin/container-admin start --detached`
  - Stop all: `bash .devcontainer/bin/container-admin stop`
  - Status: `bash .devcontainer/bin/container-admin status`
 - Persist VibeKanban Data:
  - Data directory is already bind-mounted via `/srv/devhome/.local/share/ai/bloop/vibe-kanban` for durability across container rebuilds.
- uv performance:
  - Option A: co‑locate `.venv` and `~/.cache/uv` on the same filesystem to enable hardlinks.
  - Option B: keep as is and explicitly note the expected ~5s initial cost.
- MCP enhancements:
  - Add endpoints for bulk ticket create/update to streamline migration.
  - Add a field for “reference doc path(s)” so tickets strongly link to `docs/briefs/`.
- UI polish:
  - Switch to Inter font and consider a slightly larger base font-size for readability.
  - Add a “Copy worktree path” button per ticket; low‑risk, high convenience.

Open questions for the Owner
- Do you want `.venv` and uv cache co‑located for maximum speed, or keep the current “external cache + internal venv” with the small penalty?
- Shall I switch to host bind mounts for VibeKanban app data (`~/.local/share/ai/bloop/vibe-kanban/`) and also mount `/var/tmp/vibe-kanban/worktrees` to persist worktrees?
- Preferred font: Inter, or another (e.g., IBM Plex Sans, SF Pro)?
- Do you want me to extend the `.devcontainer/bin` helpers (e.g., add a single `dev-restart.sh`) now?
- Need an importer for future bulk ticket creation (from spreadsheets, briefs, etc.)?
- Should we add a standard ticket template (fields for `why`, acceptance criteria, test steps, reference docs) and enforce it in the UI?
