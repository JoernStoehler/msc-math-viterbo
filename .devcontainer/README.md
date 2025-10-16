# Devcontainer Environment (Owner Workflow)

This directory contains the environment scaffolding for the “Project Owner” golden path. It is intentionally scoped: environment preparation here; project tasks live in the repo root.

Components
- devcontainer.json: base image + post-create/start hooks
- post-create.sh: install uv, just, VS Code CLI; quick project sync
- post-start.sh: idempotent ownership fixups and fast uv sync; prints hints
- Justfile: scoped task runner for environment services (VibeKanban, VS Code Tunnel, Cloudflared)

Daily usage (inside container)
- Start services (each in its own terminal):
  - just -f .devcontainer/Justfile start-vibe
  - just -f .devcontainer/Justfile start-tunnel
  - just -f .devcontainer/Justfile start-cf
- Status/Stop:
  - just -f .devcontainer/Justfile owner-status
  - just -f .devcontainer/Justfile owner-stop

Cloudflare Worker (font injection)
- Files under `.devcontainer/cloudflare/` (wrangler-based):
  - worker-font-injector.js, wrangler.toml
- Deploy (in container):
  - just -f .devcontainer/Justfile cf-worker-deploy
- Tail logs:
  - just -f .devcontainer/Justfile cf-worker-tail

Bind mounts (host recommended)
- Host mirrors container HOME via /srv/devhome:
  - /srv/devhome/.config/gh → /home/codespace/.config/gh
  - /srv/devhome/.vscode → /home/codespace/.vscode
  - /srv/devhome/.config/codex → /home/codespace/.config/codex
  - /srv/devhome/.cloudflared → /home/codespace/.cloudflared
  - /srv/devhome/.cache/uv → /home/codespace/.cache/uv
  - /srv/devhome/.local/share/ai/bloop/vibe-kanban → /home/codespace/.local/share/ai/bloop/vibe-kanban
- Worktrees:
  - /srv/devworktrees/vibe-kanban/worktrees → /var/tmp/vibe-kanban/worktrees

Start devcontainer on host
devcontainer up --workspace-folder /srv/workspaces/msc-math-viterbo

Container environment defaults
- Set in `.devcontainer/devcontainer.json` under `containerEnv`:
  - `FRONTEND_PORT=3000`, `HOST=0.0.0.0` (VibeKanban UI)
  - `TUNNEL_NAME=viterbo-dev` (VS Code Tunnel name)
  - `CF_TUNNEL=vibekanban` (Cloudflared tunnel name)
  - `VSCODE_CLI_DATA_DIR=/home/codespace/.vscode` (persist CLI data under the mounted path)
  - `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` pinned to HOME paths
  - You can override these per-session when running commands if needed.

Notes
- We keep .venv per worktree for isolation and accept uv’s cross-filesystem copy fallback as a small one-time cost.
- We do not auto-start services in hooks; the Justfile gives explicit, composable control.
- Upstream VibeKanban is used via npx; we don’t fork it. Font customization can be applied at the edge with a Cloudflare Worker if desired.
