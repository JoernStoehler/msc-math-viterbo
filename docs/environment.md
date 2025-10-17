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
  - sudo mkdir -p /srv/devhome/.config/gh /srv/devhome/.vscode /srv/devhome/.config/codex /srv/devhome/.config/.wrangler /srv/devhome/.cloudflared /srv/devhome/.cache/uv /srv/devhome/.local/share/ai/bloop/vibe-kanban
  - sudo mkdir -p /srv/devworktrees/vibe-kanban/worktrees
  - sudo chown -R "$USER:$USER" /srv/devhome /srv/devworktrees
- Clone repo under `/srv/workspaces/msc-math-viterbo` (preferred single path for simplicity).
- In Cloudflare Zero Trust, create the `VibeKanban Owner Board` Access application that protects `https://vibekanban.joernstoehler.com` (see `.devcontainer/README.md` for the exact choices).

One-shot host startup (recommended)
- Start everything from the host:
  - `bash .devcontainer/bin/owner-up.sh`
- What it does: brings up the devcontainer, runs preflight, starts VS Code tunnel + Cloudflared + VibeKanban (detached), then verifies.

Devcontainer (low-level alternative)
- `devcontainer up --workspace-folder /srv/workspaces/msc-math-viterbo`

Notes
- Data dir (Linux): ~/.local/share/ai/bloop/vibe-kanban (contains db.sqlite, config.json, profiles.json).
- Worktrees base: /var/tmp/vibe-kanban/worktrees.
- Cloudflare Worker: configured under `.devcontainer/cloudflare/` and deployed with wrangler via:
  - `cd .devcontainer/cloudflare && wrangler deploy`
  - Requires `wrangler login` (browser flow) in the container once.
- Keep .venv per worktree; keep uv cache central (~/.cache/uv). A small hardlink→copy fallback cost on first sync is expected and acceptable.
- post-create.sh installs uv and delegates service tooling to dev-install.sh.
- post-start.sh fixes permissions idempotently and prints diagnostics (no auto-start).

Daily start (inside the container)
- Start all (detached): `bash .devcontainer/bin/dev-start.sh --detached`
- Status: `bash .devcontainer/bin/dev-status.sh`
- Stop: `bash .devcontainer/bin/dev-stop.sh`
- Cloudflare Access: the public URL (`https://vibekanban.joernstoehler.com`) is gated by a Zero Trust Access application. Maintain the allowlist under Zero Trust → Access → Applications, and sign in with the configured IdP/OTP before using the board. Issue a service token only if non-browser automation needs to reach the board.
- Status/Stop (alternative): see the commands above.

One-shot host shutdown
- From host: `bash .devcontainer/bin/owner-down.sh` (best-effort stop in container, then `devcontainer down`, plus a non-destructive host scan)

Rebuild (host)
- Rebuild and restart the devcontainer (safe if not running):
  - `bash .devcontainer/bin/owner-rebuild.sh`
- Force a clean image build:
  - `bash .devcontainer/bin/owner-rebuild.sh --no-cache`

Host diagnostics
- Non-destructive host status: `bash .devcontainer/bin/owner-status-host.sh`

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
