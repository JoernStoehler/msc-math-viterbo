---
name: devcontainer-ops
description: Operate the project owner’s devcontainer lifecycle scripts safely during development or debugging sessions.
last-updated: 2025-10-17
---

# Devcontainer Operations

## Scope

Use this skill whenever a task involves starting, stopping, or troubleshooting the project devcontainer or its bundled services (VS Code tunnel, Cloudflared, VibeKanban).

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

## Related Skills

- `repo-onboarding` — confirms when to load this skill.
- `testing-workflow` — run after the container is operational to validate project health.
