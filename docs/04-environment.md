# Environment

The development environments are containerized. No manual bootstrap is required beyond opening the repository in the provided environment.

## Devcontainer

- Base image: `mcr.microsoft.com/vscode/devcontainers/universal:2`
- Environments:
  - Local: VS Code Dev Containers
  - Codex Cloud: configure UI entrypoints to our `.devcontainer` scripts
- Lifecycle scripts:
  - `.devcontainer/devcontainer.json`: mounts, workspace folder, extensions
  - `.devcontainer/post-create.sh`: installs Julia via Juliaup into `~/.julia/juliaup` and symlinks `julia`/`juliaup` into `~/.local/bin`
  - `.devcontainer/post-start.sh`: sets up persistent shell history and instantiates the project
- Preinstalled tooling highlights:
  - Julia via Juliaup (PATH is ensured via symlinks in `~/.local/bin`; no rc-file edits)
  - Standard CLI: `git`, `curl`, `jq`
  - Codex CLI: installed best-effort via npm as `@openai/codex` if Node/npm is present
- Mounted volumes (local): GitHub auth, Codex auth/config, persistent bash history, Julia depot (`~/.julia`)

### Verification

- `which julia` → `$HOME/.local/bin/julia`
- `julia -v` → `1.11.x`
- VS Code: “Julia: Start REPL” launches without warnings.

## GitHub

- CI at `.github/workflows/ci.yml` runs format checks (JuliaFormatter), Aqua quality checks, tests, and uploads coverage to Codecov.
- Issue templates exist for bugs, features, and RFCs; PRs use a lightweight agent-oriented checklist.

## Rationale & Decisions

- Single cache volume: persist all Julia state under `~/.julia` for fast rebuilds and simplicity.
- Official installer: use Juliaup via curl | sh with `-p` to install into `~/.julia/juliaup`. Do not rely on rc-file PATH edits; instead, post-create symlinks `~/.julia/juliaup/bin/{julia,juliaup}` into `~/.local/bin` (already on PATH) for predictable detection by terminals and the VS Code extension.
- Version policy: install Julia version matching `Project.toml` `[compat]` (currently `1.11`).
- No Dev Container Features: prefer small, readable scripts over features for predictability and speed.
