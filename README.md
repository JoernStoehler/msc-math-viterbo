# Viterbo — PyTorch + C++ Skeleton

[![Docs](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml/badge.svg)](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://joernstoehler.github.io/msc-math-viterbo)

Minimal, fast-to-iterate stack for numerical experiments around the Viterbo conjecture:
- PyTorch-first math (ragged-friendly), CPU baseline; GPU optional in models
- C++ extension scaffold for non-SIMD hotspots (CPU), with Python fallback
- Lean tests (smoke + benchmarks), incremental by default

See AGENTS.md for the authoritative policy and workflows.

## Quickstart

1) Sync deps: `just sync`
2) Fast loop: `just checks` (format → lint → type → test incremental)
3) Full local CI: `just ci` (lint → type → smoke tests → docs build)
4) Notebooks: `uv run python notebooks/dummy_generate.py`, then `uv run python notebooks/dummy_plot.py`

## Commands (common)

```
just checks      # format → lint → type → test (incremental)
just test        # smoke tests (incremental selection)
just test-full   # smoke tests (full)
just test-deep   # smoke + deep tiers
just bench       # benchmarks (smoke)
just ci          # CI parity: lint → type → tests → docs
just docs-build  # Build MkDocs site (strict)
```

## Layout

- `src/viterbo/`
  - `math/` — pure geometry/math utilities (Torch tensors I/O); no I/O, no state
  - `datasets/` — datasets + collate functions for ragged data; thin wrappers around math
  - `models/` — experiments; may use GPU; no core math here
  - `_cpp/` — C++/pybind11 extensions (CPU baseline) with safe Python fallbacks
- `tests/` — smoke tests under `test_*.py`; benchmarks under `tests/performance/`
- `docs/` — site content (see Architecture below) + tasks
- `notebooks/` — minimal examples for artefacts I/O

## Architecture (reference)

- Everyday overview lives in `AGENTS.md` (“Architecture Overview”).
- Deeper rationale and decisions: `docs/architecture/overview.md`.

## Tasks (parallel development)

- Task briefs live under `docs/tasks/` (YAML front matter + clear scope + acceptance criteria).
- Start with the open tasks in that folder; see archived examples under `docs/tasks/archived/`.

## License

MIT (`LICENSE`).
