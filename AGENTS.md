# AGENTS.md — Project Onboarding (for Codex)

This is a Julia-only project centered on the Viterbo conjecture. The environment and conventions are designed to be explicit, predictable, and agent-friendly.

## Read Me First

- Project goal: `docs/01-goal.md`
- Roadmap: `docs/02-roadmap.md`
- Conventions: `docs/03-conventions.md`
- Environment: `docs/04-environment.md`
- Working with Codex: `docs/working-with-codex.md`

## Quick Commands

Preferred (via `make`):

```bash
make setup    # Install deps
make test     # Run tests (incl. Aqua)
make format   # Format (best-effort; CI enforces via diff)
```

Fallback (raw Julia commands):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia -e 'using Pkg; Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'
```

## Environments

- Local devcontainer: open the folder and let VS Code set up the container.
- Codex Cloud: set entrypoints in the Codex UI to call our `.devcontainer` scripts:
  - Setup (one-time): `bash .devcontainer/post-create.sh`
  - Start (every boot): `bash .devcontainer/post-start.sh`

## Security

- Never log secrets. Env only. Do not echo env or use shell `-x`.

## Notes

- `Manifest.toml` is committed for reproducibility. If it drifts, regenerate in the devcontainer and commit.
- Conventions and “agent theory” are adapted from a modern Node/TS project but tailored to Julia.

## Project Preferences

- Formatting: JuliaFormatter with `indent = 2` (see `.JuliaFormatter.toml`).
- Manifest: Commit `Manifest.toml` and keep it in sync.
- Docs: Markdown only; no Documenter.jl site.
- CI/README: No badges in `README.md`.
- Analysis: Do not use JET in CI (too slow for our workflow).
- Contributing: No `CONTRIBUTING.md` (use this `AGENTS.md`).
- Platform: Linux-only support assumptions.
