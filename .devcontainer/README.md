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

Cloudflare Access (one-time, host-side)
---------------------------------------
Goal: require authentication before anyone reaches `https://vibekanban.joernstoehler.com`. Follow Cloudflare’s official guide (“[Create an Access application](https://developers.cloudflare.com/learning-paths/clientless-access/access-application/create-access-app/)”) with the concrete choices below so naming stays consistent.

### Prerequisites
- Cloudflare Zero Trust account with the `joernstoehler.com` zone already connected.
- Identity provider ready for Access (e.g. “One-time PIN” email IdP or GitHub). Keep it simple to start; you can layer more providers later.

### Step-by-step (Zero Trust dashboard)
1. Sign in at <https://one.dash.cloudflare.com> using the account that owns the zone.
2. Navigate to **Zero Trust** → **Access** → **Applications**.
3. Click **Add an application** → choose **Self-hosted**.
4. Fill out the application card:
   - **Name**: `VibeKanban Owner Board`
   - **Domain**: select `vibekanban.joernstoehler.com`
   - **Session Duration**: `1w` (Cloudflare’s max; adjust downward if you prefer more frequent re-auth).
   - Leave the rest at defaults unless you have a reason to change them (no browser rendering needed for a standard web UI).
5. Select **Next** to configure policies, then add a single **Allow** policy named `Owner-only`:
   - **Action**: *Allow*
   - **Include** rule: pick your primary auth method. For a minimal setup choose **Emails** and list the addresses that should reach the board (e.g. `joern@joernstoehler.com`). Alternatively, select the GitHub IdP and point at your organization if you already use it.
   - You can leave **Exclude** and **Require** empty for now.
6. (Optional) Add a **Service Auth** policy if you ever need automation. Give it a descriptive name (e.g. `ci-service-token`), set **Decision** to *Allow*, and choose **Service Tokens** → **Create new token**. Store the generated client ID/secret somewhere safe.
7. Click **Next**, skip App Launcher/block page customisation, then **Save**.
8. Test in a clean browser (or private window) on each device (Android tablet, university Windows PC). You should hit the Cloudflare Access login first, complete the OTP/IdP flow, then reach VibeKanban.

### Maintenance
- Whenever you on-board someone else, revisit **Zero Trust** → **Access** → **Applications** → `VibeKanban Owner Board` → **Policies** and extend the allowlist instead of cloning the app.
- If you revoke access, remove the email/org or delete the service token so old sessions expire naturally at the next validation.
- Record any additions/removals in `mail/` weekly status notes so the audit trail stays outside the dashboard.

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
