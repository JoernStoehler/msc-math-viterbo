# Devcontainer Environment (Owner Workflow)

This directory contains the environment scaffolding for the “Project Owner” golden path. It is intentionally scoped: environment preparation lives here; project tasks stay in the repo root.

Components
- `devcontainer.json`: base image + post-create/post-start hooks
- `post-create.sh`: install uv, just, VS Code CLI; quick project sync
- `post-start.sh`: idempotent ownership fixups, fast uv sync, diagnostics
- `Justfile`: scoped task runner for environment services (VibeKanban, VS Code Tunnel, Cloudflared)

Daily usage (inside container)
- Start services (one terminal per command):
  - `just -f .devcontainer/Justfile start-vibe`
  - `just -f .devcontainer/Justfile start-tunnel`
  - `just -f .devcontainer/Justfile start-cf`
- Status / stop:
  - `just -f .devcontainer/Justfile owner-status`
  - `just -f .devcontainer/Justfile owner-stop`

Cloudflare Worker (font injection)
- Files under `.devcontainer/cloudflare/` (wrangler-based)
  - `worker-font-injector.js`, `wrangler.toml`
- Deploy in container: `just -f .devcontainer/Justfile cf-worker-deploy`
- Tail logs: `just -f .devcontainer/Justfile cf-worker-tail`

Cloudflare tunnel (one-time prep)
- `just -f .devcontainer/Justfile owner-cloudflare-setup` — write `config-<tunnel>.yml` and ensure the DNS route points at the tunnel (requires Cloudflare login + tunnel auth).

Bind mounts (host recommended)
- Host mirrors container HOME via `/srv/devhome`:
  - `/srv/devhome/.config/gh` → `/home/codespace/.config/gh`
  - `/srv/devhome/.vscode` → `/home/codespace/.vscode`
  - `/srv/devhome/.config/codex` → `/home/codespace/.config/codex`
  - `/srv/devhome/.config/.wrangler` → `/home/codespace/.config/.wrangler` (persists Cloudflare Wrangler OAuth tokens)
  - `/srv/devhome/.cloudflared` → `/home/codespace/.cloudflared` (tunnel certs + config)
  - `/srv/devhome/.cache/uv` → `/home/codespace/.cache/uv`
  - `/srv/devhome/.local/share/ai/bloop/vibe-kanban` → `/home/codespace/.local/share/ai/bloop/vibe-kanban`
- Worktrees:
  - `/srv/devworktrees/vibe-kanban/worktrees` → `/var/tmp/vibe-kanban/worktrees`

Cloudflared installation (deterministic)
---------------------------------------
- `post-create.sh` installs `cloudflared` from Cloudflare’s official APT repo (Ubuntu 20.04 / focal).
- Steps: write the key to `/usr/share/keyrings/cloudflare-main.gpg`, add `https://pkg.cloudflare.com/cloudflared focal main`, run `apt-get update`, then `apt-get install -y cloudflared`.
- Failures are surfaced explicitly so you can fix the image or install manually. Adjust the suite (e.g. jammy) if you change the base image.
- After installing and authenticating, run `just -f .devcontainer/Justfile owner-cloudflare-setup` once to wire the hostname to the tunnel.

Rebuild / test the environment
- Rebuild the devcontainer (Reopen/Rebuild in Container) and review the post-create logs.
- Or run inside the container:
  ```bash
  sudo apt-get update
  bash .devcontainer/post-create.sh
  ```
- Confirm `cloudflared` installs cleanly or follow the printed error.

Start devcontainer on host
`devcontainer up --workspace-folder /srv/workspaces/msc-math-viterbo`

Container environment defaults
- Defined under `containerEnv` in `devcontainer.json`:
  - `FRONTEND_PORT=3000`, `HOST=0.0.0.0` (VibeKanban UI)
  - `TUNNEL_NAME=viterbo-dev` (VS Code Tunnel name)
  - `CF_TUNNEL=vibekanban` (Cloudflared named tunnel)
  - `VSCODE_CLI_DATA_DIR=/home/codespace/.vscode` (aligns with the mount)
  - `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` pinned to HOME paths
- Override per session if needed when running commands.

Notes
- Keep `.venv` per worktree for isolation; uv handles cross-filesystem copies once.
- Hooks never auto-start services; the Justfile gives explicit control.
- VibeKanban ships via `npx`; font customization happens via the Cloudflare Worker.
