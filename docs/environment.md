# Environment (Owner Workflow)

This page documents the “golden path” environment used by the Project Owner. It is stable, explicit, and minimal — agents don’t need to understand every piece to do their day-to-day work; they can refer here for maintenance or when refactoring the environment.

Overview
- Client: Chrome on Android tablet.
- Host workstation: Ubuntu 24.04 desktop with Tailscale.
- Devcontainer: the project’s full dev stack — VibeKanban (via npx), agents, repo, VS Code Tunnel, Cloudflared.
- Persistence: host bind mounts for tokens, caches, VibeKanban data, and worktrees.

Host (one-time)
- Install Tailscale; log in.
- Install devcontainer CLI.
- Create bind-mount roots (example):
  - sudo mkdir -p /srv/devhome/.config/gh /srv/devhome/.vscode /srv/devhome/.config/codex /srv/devhome/.cloudflared /srv/devhome/.cache/uv /srv/devhome/.local/share/ai/bloop/vibe-kanban
  - sudo mkdir -p /srv/devworktrees/vibe-kanban/worktrees
  - sudo chown -R "$USER:$USER" /srv/devhome /srv/devworktrees
- Clone repo under `/srv/workspaces/msc-math-viterbo` (preferred single path for simplicity).

Devcontainer (start on host)
devcontainer up --workspace-folder /srv/workspaces/msc-math-viterbo

Notes
- Data dir (Linux): ~/.local/share/ai/bloop/vibe-kanban (contains db.sqlite, config.json, profiles.json).
- Worktrees base: /var/tmp/vibe-kanban/worktrees.
- Cloudflare Worker: configured under `.devcontainer/cloudflare/` and deployed with wrangler via:
  - `just -f .devcontainer/Justfile cf-worker-deploy`
  - Requires `wrangler login` (browser flow) in the container once.
- Keep .venv per worktree; keep uv cache central (~/.cache/uv). A small hardlink→copy fallback cost on first sync is expected and acceptable.
- post-create.sh installs uv, just, VS Code CLI and performs a light uv sync.
- post-start.sh fixes permissions idempotently and prints diagnostics; it does not auto-start services.

Daily start (inside the container)
- Start services (each in its own terminal):
  - VibeKanban: just -f .devcontainer/Justfile start-vibe
  - VS Code Tunnel: just -f .devcontainer/Justfile start-tunnel
  - Cloudflared: just -f .devcontainer/Justfile start-cf
- Status/Stop:
  - just -f .devcontainer/Justfile owner-status
  - just -f .devcontainer/Justfile owner-stop

Client (Android Chrome)
- Open VS Code Tunnel URL and https://vibekanban.joernstoehler.com
- To open a task worktree: run `code --add /var/tmp/vibe-kanban/worktrees/<id>` in the devcontainer shell.

Font customization (optional)
- Keep upstream VibeKanban pristine and inject “Inter” at the edge with a simple Cloudflare Worker on vibekanban.joernstoehler.com/* using HTMLRewriter to append the <link> and <style> tags into <head>. This isolates UI tweaks from upstream releases.

Auth hints (first time after switching to bind mounts)
- gh: gh auth login
- VS Code tunnel: code tunnel (will guide through auth)
- cloudflared: cloudflared tunnel login
- codex CLI (if used): re-auth if necessary

Golden-path stance
- No Codespaces; no Codex Cloud. We run locally on the workstation via devcontainer.
- Services are independent; start/stop individually for resilience.
- VibeKanban via npx (no fork). Persist its data and worktrees on host bind mounts.
