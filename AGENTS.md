# AGENTS.md

Single authoritative policy for this repo.

## 0) Roles & Scope

- Maintainer (PI)
  - Approves task briefs and larger directional changes.
  - Owns DevOps/CI, merges PRs, and steers research/architecture.
- Codex agents (ephemeral)
  - Implement focused, incremental changes (task level).
  - Escalate uncertainties early.
  - Open PRs and iterate until CI is green; PI merges.
- Escalation triggers (PR description with `Needs-Unblock: <topic>` or issue)
  - Ambiguous acceptance criteria; policy conflicts; larger env/CI changes.
  - Cross‑task architecture choices; performance regressions beyond thresholds.

## 1) Sources of Truth & Layout

- AGENTS.md (this file, read first)
- MIGRATION.md (PyTorch+C++ migration log)
- Config: `pyproject.toml` (deps, Ruff), `pyrightconfig.json` (basic), `pytest.ini` (smoke defaults), `.github/workflows/ci.yml` (CI), `.devcontainer/` (environment)
- Task runner: `Justfile` (common commands)
- Library: `src/viterbo/`
  - `math/` — pure geometry/math utilities (Torch tensors I/O). No I/O, no state.
  - `datasets/` — adapters/datasets/collate for ragged data; thin wrappers around math.
  - `models/` — experiments/training loops; may use GPU; no core math here.
  - `_cpp/` — C++/pybind11 extensions (CPU baseline) with Python fallbacks.
- Tests: `tests/` — smoke tests under `test_*.py`; benches under `tests/performance/`
- Docs & Notes: `docs/` (site + task briefs), `notebooks/` (dummy examples), `artefacts/` (outputs, ignored)
  - Tasks: `docs/tasks/` (open briefs) and `docs/tasks/archived/` (examples)
  - `notebooks/` stores Jupytext-managed `.py` notebooks; preserve the front-matter metadata when editing or adding entries.

## 2) Environment & Tooling

- Stack: Python 3.12, PyTorch 2.x (CPU baseline; optional CUDA for models only). C++17 with pybind11 for custom hotspot non‑SIMD kernels.
- Devcontainers: `.devcontainer/devcontainer.local.json` (local VS Code Remote - Containers with host volumes) and `.devcontainer/devcontainer.codespaces.json` (GitHub Codespaces without host volumes). Pick the correct definition when prompted; there is no default.
- Codex agents land inside a pre-provisioned environment; never run devcontainer lifecycle commands manually.
- Dependency manager: uv (`uv run`, `uv sync`, `uv add`). Commit `uv.lock`.
- Editors: Pyright (basic) for fast feedback; Ruff for lint/format.
- Testing: Pytest (smoke by default) + incremental selector (`scripts/inc_select.py`) for fast local loops + `pytest-benchmark` for targeted benches.

### Quick Commands

- `just checks` - for quick feedback, runs `just format && just lint && just type && just test`
- `just fix` - auto-fix formatting/linting issues
- `just test` - incremental smoke tests (fast)
- `just bench` - smoke benchmarks (saves under `.benchmarks/`)
- `just ci` - CI parity, non-incremental test run, pass before pushing/PR

## 3) Coding Conventions (facts)

- PyTorch‑first: library code uses `torch` tensors; return tensors from public APIs.
- Precision: set dtype per function/docstring (math often float64; ML often float32). Avoid silent downcasts; document deviations.
- Ragged data: allow Python lists of tensors or padded tensors with masks; expose `collate_fn`s in `datasets`.
- Purity: `viterbo.math` is pure (no I/O, no hidden state). Keep side‑effects in adapters.
- Docstrings: concise Google‑style focusing on semantics, invariants, units and shapes. Add shape/dtype comments where non‑obvious.
- Imports & structure: absolute imports with explicit paths; no wildcard imports; No re‑export indirection; namespaced modules (no `__all__`).
- Types: prefer built‑ins (`list[str]`, `dict[str, torch.Tensor]`, unions with `|`); avoid custom typedefs for shapes/dimensions.
- Commit: Conventional Commits.
- Placeholders: do **not** wrap `NotImplementedError` (or other TODO sentinels) in `try/except`; allow the error to surface so missing implementations remain obvious during TDD loops.
- Assertions: if an `assert` passes during development, do not duplicate it purely to survive `python -O`.

## 4) PyTorch + C++ specifics (facts)

- Device policy: math APIs accept tensors on caller’s device; no implicit device moves.
- RNG: prefer `torch.Generator` handles over bare integer seeds; pass seeds explicitly only when bridging external APIs.
- Runtime: assume CPU-only execution unless a task explicitly calls for CUDA.
- C++: use `torch.utils.cpp_extension` + pybind11; start CPU‑only. Add CUDA only when required.
- Plotting/IO: push conversions to call sites (e.g., `tensor.detach().cpu().numpy()` when needed).

## 5) Minimal Example (PyTorch)

```python
import torch

def support(points, direction):
    """Support function of a finite point set.

    Args:
      points: (N, D) float64
      direction: (D,) float64, not normalized

    Returns:
      () float64 tensor:
        max_i <points[i], direction>
    """
    return (points @ direction).max()
```

## 6) Testing (facts)

- Keep tests pragmatic and fast. Prefer smoke tests and representative benchmarks.
- Structure: organize by module (`tests/test_*.py`, `tests/performance/test_*.py`).
- Timeouts: keep smoke tests under a few seconds locally and in CI.
- Benchmarks: use `pytest-benchmark` with fixed RNG seeds; save artefacts under `.benchmarks/`.
- Assertions: use `pytest.approx`, `torch.testing.assert_close`, or `math.isclose` appropriately.
- Shape validation: rely on docstrings, inline comments, and the existing unit tests; avoid redundant shape assertions unless a bug fix demands them.

## 7) Performance (facts)

- Start with pure Python/Torch; introduce C++ for clear hotspots only.
- Benchmarks live in `tests/performance/`; use `just bench` to run smoke benches.
- Profile on demand with local tools; notebooks are fine for exploration.

## 8) Workflows (imperative)

Daily development

1. Read task; scan relevant code/tests.
2. Reflect on goal and develop a short plan (4–7 steps).
3. Implement cohesive changes. Run `just precommit` for local gate (format, lint, type, smoke).
4. Keep math pure; do I/O only in datasets/models/adapters.
5. For parity, run `just ci-cpu` locally if needed; update tests/docs before pushing / opening PR.

PR message:

- Feature changes, scope, files touched, how tested (Ruff/Pyright/pytest), perf delta if relevant, limitations and follow‑ups.

## 9) Conventions

- Use semantic names like `normals`, `offsets` for halfspaces.
- Avoid dataclasses in `math`; return tensors/tuples of tensors.
- Keep `datasets` simple; no DSL; explicit functions and small classes.

## 12) Policy Notes (imperative)

- Strict layering: `math` must not depend on `datasets`/`models`.
- No `__all__`; avoid re‑export indirection (no aggregator re‑exports in `__init__.py`).
- Prefer small, focused cleanups over exceptions.

## 13) Architecture Overview (everyday reference)

- Layering: `math` (pure, stateless) ← `datasets` (adapters, ragged collation) ← `models` (experiments; may use GPU). C++ kernels live under `_cpp` with Python fallbacks.
- Ragged patterns: use lists of tensors or padded tensors + masks; provide collate functions (`collate_list`, `collate_pad`).
- Devices: accept caller’s device; do not move implicitly. Dtypes are documented at each function.
- C++ interop: CPU‑only baseline via `torch.utils.cpp_extension`; no CUDA unless required; keep fallbacks to Python/Torch.

## 14) Current Focus

- Primary research target: 4D polytopes in the symplectic standard setting.
