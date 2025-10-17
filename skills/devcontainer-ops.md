---
name: devcontainer-ops
description: This skill should be used when starting, stopping, or troubleshooting the devcontainer and its services.
last-updated: 2025-10-17
---

# Devcontainer Operations

## Instructions
- Start the environment with `bash .devcontainer/bin/owner-up.sh`; capture errors and outputs in task notes if anything fails.
- Manage in-container services via `.devcontainer/bin/dev-*.sh` and shut down with `bash .devcontainer/bin/owner-down.sh`.
- Use `bash .devcontainer/bin/owner-status-host.sh` for safe status checks; avoid partial manual restarts.
- Escalate with `Needs-Unblock: devcontainer` for recurring lifecycle issues or script changes.

## Scope

Covers starting, stopping, and troubleshooting the project devcontainer and bundled services (VS Code tunnel, Cloudflared, VibeKanban).

## Host-Level Lifecycle

1. To start services: run `bash .devcontainer/bin/owner-up.sh`.
   - Confirms prerequisites, starts the container, launches tunnels, and surfaces status output.
2. To stop services cleanly: run `bash .devcontainer/bin/owner-down.sh`.
   - Wait for completion before restarting to avoid orphaned tunnels.
3. To rebuild the container after dependency or base image updates: run `bash .devcontainer/bin/owner-rebuild.sh`.
4. For a non-destructive status check: run `bash .devcontainer/bin/owner-status-host.sh`.

## In-Container Controls

- Start development processes inside the container with `.devcontainer/bin/dev-start.sh`.
- Check running services via `.devcontainer/bin/dev-status.sh`.
- Shut down services gracefully using `.devcontainer/bin/dev-stop.sh`.

## Safety Guidelines

- Avoid partial manual restarts; always use the paired start/stop scripts to keep tunnels consistent.
- Log script output (copy critical lines into task notes) when lifecycle changes occur.
- If scripts fail, inspect recent changes to `.devcontainer/bin/` before retrying; escalate if errors persist.
- Never edit `.devcontainer/bin/` scripts directly unless the maintainer assigns a task covering those files.
- Reference `docs/environment.md` when host prerequisites differ from the golden path (e.g., missing packages or OS updates).
- Keep `uv`-managed caches intact; rely on the scripts to handle environment synchronization instead of manual pip installs.

## Troubleshooting Checklist

1. Capture the full command output and exit code.
2. Confirm whether another agent already has services running to avoid double-start conflicts.
3. Verify network constraints if tunnels fail—document firewall or VPN interactions in task notes.
4. After resolving an issue, rerun `owner-status-host.sh` to confirm steady state before handing off.

## Related Skills

- `repo-onboarding` — confirms when to load this skill.
- `testing-workflow` — run after the container is operational to validate project health.
- `collaboration-reporting` — document lifecycle incidents that impact other collaborators.
