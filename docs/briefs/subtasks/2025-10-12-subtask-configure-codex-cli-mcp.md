---
status: draft
created: 2025-10-12
workflow: task
summary: Draft the steps to configure Codex CLI with the Datalayers Jupyter MCP server during post-start.
---

# Subtask: Configure Codex CLI for Datalayers Jupyter MCP

## Context

- `.devcontainer/post-start.sh` handles recurring bootstrapping (history persistence, uv sync, PATH tweaks).
- Codex CLI currently runs in stateless mode; without MCP it cannot orchestrate persistent Jupyter kernels or display cell outputs.
- Datalayers ships an MCP server that should bridge Codex CLI to a Jupyter kernel for notebook authoring.

## Objectives (initial draft)

- Ensure `post-start.sh` idempotently installs/configures the Codex CLI MCP plugin targeting the Datalayers Jupyter server.
- Verify Codex CLI can start, restart, and run notebook cells with persistent state within the container.
- Provide diagnostics/logging to help future contributors validate the integration.

## Deliverables (tentative)

- Updated `.devcontainer/post-start.sh` (and supporting scripts/config) performing the MCP setup.
- Documentation snippet (README or workflow brief) describing how to use the MCP-enabled Codex CLI.
- Smoke test or command sequence proving notebook execution works headlessly.

## Dependencies

- Uses the existing devcontainer lifecycle hooks (`.devcontainer/post-start.sh`).
- Requires Codex CLI binaries to be present (installed via existing tooling) and network access to fetch MCP plugins if necessary.
- Integration quality is validated by notebook-centric subtasks that rely on persistent kernels.

## Acceptance criteria (to validate completion)

- Post-start provisioning is idempotent: repeated container launches do not duplicate configuration or break Codex CLI usage.
- Running `codex --list-mcp` (or equivalent) shows the Datalayers Jupyter MCP server registered and available.
- Documentation includes authentication guidance, environment variables, and troubleshooting steps for common failures.
- A scripted or documented smoke test demonstrates executing a notebook cell via Codex CLI with observable persisted state across runs.

## Decisions and constraints

- Installation/configuration should follow the upstream docs (`datalayer/jupyter-mcp-server` and Codex CLI guidance) and likely reduces to appending the provided snippet to `~/.codex/config.toml`; Codex handles plugin acquisition afterward.
- Scope: limit automation to local devcontainer environments; Codex Cloud lacks MCP support for now.
- Post-start does not need an automated verification step—success can be confirmed manually when Codex CLI reports the loaded MCP servers.

## Open Questions

1. What authentication or API keys are required, and how should we source them securely?
2. Are there version constraints between Codex CLI and the MCP server we must respect?
3. Do we need to register additional MCP tools (filesystem, git) alongside Jupyter?

## Notes

- Capture any required secrets or auth steps in `.env.example` or similar templates rather than hardcoding into scripts.
