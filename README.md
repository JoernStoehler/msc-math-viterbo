# Viterbo Conjecture — Julia Project

This repository explores the Viterbo conjecture and related symplectic capacities using Julia. It is a single-language, single-project setup with a streamlined developer experience for both Local Dev Containers and Codex Cloud.

## Getting Started

- Install dependencies:
  `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Run tests:
  `julia --project=. -e 'using Pkg; Pkg.test()'`
- Optional (format):
  `julia -e 'using Pkg; Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'`

## Repository Layout

- `src/`: Julia source (module `ViterboConjecture`)
- `test/`: Unit tests (`Test` stdlib) and Aqua quality checks
- `docs/`: Project goal, roadmap, conventions, environment, and Codex collaboration notes
- `.devcontainer/`: Devcontainer config and lifecycle scripts (Local + Codex Cloud)
- `.github/`: CI (format check, tests, coverage) and issue/PR templates

## CI

GitHub Actions runs format checks (JuliaFormatter), Aqua quality checks, unit tests, and uploads coverage to Codecov. See `.github/workflows/ci.yml`.

## License

MIT License — see `LICENSE`.

