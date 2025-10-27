# Viterbo — PyTorch + C++ Skeleton

[![Docs](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml/badge.svg)](https://github.com/JoernStoehler/msc-math-viterbo/actions/workflows/docs.yml)
[![Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://joernstoehler.github.io/msc-math-viterbo)

Minimal, fast-to-iterate stack for numerical experiments around the Viterbo conjecture:
- PyTorch-first math (ragged-friendly), CPU baseline; GPU optional in models
- C++ extension scaffold for non-SIMD hotspots (CPU). Native builds are required; there is no Python fallback.
- Lean tests (smoke + benchmarks), full smoke by default with optional selector

Start here: [`docs/milestones/current.md`](docs/milestones/current.md) for the current scope, then `AGENTS.md` for policies and workflows.

## Quickstart

0) Ensure [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is available (pipx install uv, standalone binary, or your package manager) and install a native toolchain: `build-essential` (or platform equivalent) and `ninja` must be on `PATH`.
1) Sync deps: `just sync`
2) Fast loop: `just checks` (format → lint → type → smoke tests; set `JUST_USE_SELECTOR=1` for incremental)
3) Full local CI: `just ci` (lint → type → smoke tests → docs build)
4) Notebooks: `uv run python notebooks/dummy_generate.py`, then `uv run python notebooks/dummy_plot.py`

### Compute the systolic ratio (literature normalization)

```
uv run python - << 'PY'
import torch
from viterbo.math.constructions import counterexample_pentagon_product
from viterbo.math.volume import volume
from viterbo.math.capacity_ehz.algorithms import capacity_ehz_algorithm2
from viterbo.math.capacity_ehz.ratios import systolic_ratio

torch.set_default_dtype(torch.float64)
v, n, c = counterexample_pentagon_product()
vol = volume(v)
cap = capacity_ehz_algorithm2(v)  # fast Minkowski billiards path for the 5×5 product
sys = systolic_ratio(vol, cap, 4)  # Sys = c^n / (n! Vol) with n=2 in 4D
print("volume:", float(vol.item()))
print("capacity:", float(cap.item()))
print("Sys (c^n/(n! Vol)):", float(sys.item()))
PY
```

### C++ extensions (no fallback)

- Export `USE_NINJA=1` so builds always use Ninja; `just` recipes do this for you, mirror it in ad-hoc shells.
- Keep `VITERBO_CPP_VERBOSE=1` to log compile commands; it helps surface toolchain drift.
- Core ops import `viterbo._cpp.*` on startup; failures raise immediately. Fix toolchain issues rather than disabling C++.
- Common fixes:
  - Missing compilers → install `build-essential` (Debian/Ubuntu) or the platform’s Clang/GCC toolchain.
  - `ninja: not found` → install `ninja` (package manager) or `uv add ninja`.
  - Stale artefacts → `rm -rf ~/.cache/torch_extensions/*` and retry.
  - Starved builds → set `MAX_JOBS=<n>` (e.g., `MAX_JOBS=4`) to cap parallel compilation.

## Development containers

The repository ships two development container definitions in `.devcontainer/`. VS Code and GitHub Codespaces will prompt you to
choose a configuration; there is no default on purpose.

- `devcontainer.local.json` — Local VS Code Remote - Containers workflow with host bind mounts/volumes for auth, history, and
  caches. Use when running Docker on your workstation.
- `devcontainer.codespaces.json` — GitHub Codespaces variant without host volumes. Select when creating or reopening a Codespace.

## Commands (common)

```
just checks      # format → lint → type → smoke tests (full by default; set JUST_USE_SELECTOR=1 for incremental)
just test        # smoke tests (full; opt-in to selector via JUST_USE_SELECTOR=1)
just test-fast   # smoke tests (incremental selector)
just test-full   # smoke tests (full, explicit)
just test-deep   # smoke + deep tiers
just bench       # benchmarks (smoke)
just ci          # CI parity: lint → type → tests → docs
just docs-build  # Build MkDocs site (strict)
```

Set `JUST_USE_SELECTOR=1 just test` (or run `just test-fast`) to reuse the incremental selector when you need the shorter loop.

## Layout

- `src/viterbo/`
  - `math/` — pure geometry/math utilities (Torch tensors I/O); no I/O, no state
  - `datasets/` — datasets + collate functions for ragged data; thin wrappers around math
  - `models/` — experiments; may use GPU; no core math here
  - `_cpp/` — C++/pybind11 extensions (CPU baseline). Builds are mandatory; no Python fallbacks remain.
- `tests/` — smoke tests under `test_*.py`; benchmarks under `tests/performance/`
- `docs/` — site content (architecture, math docs, workflows, briefs)
- `notebooks/` — minimal examples for artefacts I/O

## Architecture (reference)

- Everyday overview lives in `AGENTS.md` (“Architecture Overview”).
- Deeper rationale and decisions: `docs/architecture/overview.md`.
- Project Charter (goals, milestones, success metrics): `docs/charter.md`.

## Task Management

- Use the VibeKanban project `Msc Math Viterbo` as the single backlog; adopt tickets there when starting work.
- Supporting context (workflows, briefs, references) lives in `docs/` and is linked from tickets when relevant.

## License

MIT (`LICENSE`).
