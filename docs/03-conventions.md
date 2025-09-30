# Conventions

## Tech Stack

- Language: Julia 1.11 (via juliaup)
- Testing: `Test` stdlib, Aqua.jl for quality checks (no JET in CI)
- Formatting: JuliaFormatter.jl (enforced in CI) with `indent = 2`
- Documentation: Markdown in `docs/` (markdown-only; no Documenter site)
- CI: GitHub Actions (format, test, coverage)
- Dev Environments: Local devcontainer, Codex Cloud entrypoints (Linux-only assumption)

## Folder Structure

- `README.md`: Human-facing overview
- `AGENTS.md`: Codex onboarding with quick commands and environment notes
- `docs/`: Goal, roadmap, conventions, environment, working-with-codex
- `src/`: Julia source (module `ViterboConjecture`)
- `test/`: Unit tests + Aqua quality checks
- `.devcontainer/`: Devcontainer config and lifecycle scripts
- `.github/`: Workflows and templates

## Coding Conventions

- Prefer small, composable, pure functions for the mathematical core.
- Keep side effects shallow and explicit; document why where nontrivial.
- Add docstrings to public APIs; keep examples runnable where practical.
- Use explicit types only when they clarify intent or performance.
- Write tests for new helpers; colocate simple examples as doctests where suitable.

## Project Preferences

- Keep `Manifest.toml` committed; regenerate in devcontainer when needed.
- No README badges.
- Use `AGENTS.md` instead of a separate `CONTRIBUTING.md`.
- No CODEOWNERS file (ownership remains implicit in this phase).

## Docs & Writing

- Be explicit and precise; spell out assumptions and implications.
- Prefer common documentation formats (README, RFC, roadmap) for predictable navigation.
- Use short imperative instructions for developer actions.

## Communication Conventions

- Lead with the most important information; follow with supporting details.
- Use clear checklists in PRs; keep acceptance criteria testable.
- For Codex agents, honour short-form instructions when present:
  - READ ONLY — investigation/explanation only
  - IMMEDIATE — act quickly, minimal commands, respond once confident
  - TALK ONLY — reason in chat without executing commands or opening files
