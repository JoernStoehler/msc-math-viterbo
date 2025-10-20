---
name: devcontainer-ops
description: This skill should be used when starting, stopping, or troubleshooting the devcontainer and its services.
last-updated: 2025-10-18
---

# Devcontainer Operations

## Instructions
- Host orchestration: `bash .devcontainer/bin/host-admin up preflight start` (add `--interactive` to attach tmux with tips).
- In-container controls: `bash .devcontainer/bin/container-admin start|status|stop|restart`.
- Status (concise): `bash .devcontainer/bin/host-admin status` (add `--verbose` for diagnostics).
- Escalate with `Needs-Unblock: devcontainer` for recurring lifecycle issues or script changes.

## Scope

Covers starting, stopping, and troubleshooting the project devcontainer and bundled services (VS Code tunnel, Cloudflared, VibeKanban).

## Host-Level Lifecycle

1. Start: `bash .devcontainer/bin/host-admin up preflight start --interactive`.
2. Stop: `bash .devcontainer/bin/host-admin down`.
3. Rebuild: `bash .devcontainer/bin/host-admin rebuild [--no-cache]`.
4. Status: `bash .devcontainer/bin/host-admin status [--verbose]`.
5. Restart VibeKanban (hot fix): `bash .devcontainer/bin/host-admin restart`.

## In-Container Controls

- Start: `.devcontainer/bin/container-admin start --detached`.
- Status: `.devcontainer/bin/container-admin status [--verbose]`.
- Stop: `.devcontainer/bin/container-admin stop`.
- Restart VibeKanban only (hot fix): `.devcontainer/bin/container-admin restart`.

## Safety Guidelines

- Avoid partial manual restarts; always use the paired start/stop scripts to keep tunnels consistent. Exception: use `restart` for VibeKanban-only hot fix.
- Log script output (copy critical lines into task notes) when lifecycle changes occur.
- If scripts fail, inspect recent changes to `.devcontainer/bin/` before retrying; escalate if errors persist.
- Never edit `.devcontainer/bin/` scripts directly unless the maintainer assigns a task covering those files.
- Reference `docs/environment.md` when host prerequisites differ from the golden path (e.g., missing packages or OS updates).
- Keep `uv`-managed caches intact; rely on the scripts to handle environment synchronization instead of manual pip installs.

## Troubleshooting Checklist

1. Capture the full command output and exit code.
2. Confirm whether another agent already has services running to avoid double-start conflicts.
3. Verify network constraints if tunnels fail—document firewall or VPN interactions in task notes.
4. After resolving an issue, rerun `host-admin status --verbose` to confirm steady state before handing off.

## Related Skills

- `repo-onboarding` — confirms when to load this skill.
- `testing-and-ci` — run after the container is operational to validate project health.
- `collaboration-reporting` — document lifecycle incidents that impact other collaborators.
