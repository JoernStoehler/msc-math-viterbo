# AGENTS.md

Single authoritative policy for this repo. If another doc conflicts, follow AGENTS.md.

## 0) Roles & Scope (facts)

- Maintainer (PI)
  - Approves task briefs and larger directional changes.
  - Owns DevOps/CI, merges PRs, and steers research/architecture.
- Codex agents (ephemeral)
  - Implement focused, incremental changes (feature/fix/refactor/docs/tests/benchmarks).
  - Draft briefs when helpful; escalate uncertainties early.
  - Open PRs and iterate until CI is green; PI merges.
- Escalation triggers (PR description with `Needs-Unblock: <topic>` or issue)
  - Ambiguous acceptance criteria; policy conflicts; larger env/CI changes.
  - Cross‑task architecture choices; performance regressions beyond thresholds.

## 1) Sources of Truth (facts)

- AGENTS.md (this file), MIGRATION.md (PyTorch+C++ migration log).
- Config: `pyproject.toml` (deps, Ruff), `pyrightconfig.json` (basic), `pytest.ini` (smoke defaults), `.github/workflows/ci.yml` (CI), `.devcontainer/`.
- Task runner: `Justfile`.
- Waivers: `waivers.toml`.
- Code: `src/viterbo/` (library), `tests/` (smoke + benchmarks).
- Notes: `docs/` stubs and `notebooks/` examples during migration.

## 2) Environment & Tooling

- Stack: Python 3.12, PyTorch 2.x (CPU baseline; optional CUDA for models only).
- Dependency manager: uv (`uv run`, `uv sync`, `uv add`). Commit `uv.lock`.
- Editors: Pyright (basic) for fast feedback; Ruff for lint/format.

## 3) Coding Conventions (facts)

- PyTorch‑first: library code uses `torch` tensors; return tensors from public APIs.
- Precision: default to float32/float64 as context requires; avoid silent downcasts; document deviations.
- Ragged data: allow Python lists of tensors or padded tensors with masks; expose `collate_fn`s in `datasets`.
- Purity: `viterbo.math` is pure (no I/O, no hidden state). Keep side‑effects in adapters.
- Docstrings: concise Google‑style focusing on semantics, invariants, and units. Avoid over‑specifying shapes; add shape/dtype notes when useful.
- Imports & structure: absolute imports; no wildcard imports; avoid re‑export indirection; prefer explicit paths.
- Security: never print/log secrets.
- Branching: `feat/<scope>`, `fix/<scope>`, `refactor/<scope>`; Conventional Commits.

## 4) PyTorch + C++ specifics (facts)

- Device policy: math APIs accept tensors on caller’s device; no implicit device moves.
- RNG: use `torch.Generator` or seed ints passed explicitly.
- C++: use `torch.utils.cpp_extension` + pybind11; start CPU‑only. Add CUDA only when required.
- Plotting/IO: push conversions to call sites (e.g., `tensor.detach().cpu().numpy()` when needed).

## 5) Minimal Example (PyTorch)

```python
import torch

def support(points: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Support function of a finite point set.

    Args:
      points: (N, D) float tensor.
      direction: (D,) float tensor; not normalized.

    Returns:
      Scalar tensor: max_i <points[i], direction>.
    """
    return (points @ direction).max()
```

## 6) Testing (facts)

- Keep tests pragmatic and fast. Prefer smoke tests and representative benchmarks.
- Structure: organize by module (`tests/test_*.py`, `tests/performance/test_*.py`).
- Timeouts: keep smoke tests under a few seconds locally and in CI.
- Benchmarks: use `pytest-benchmark` with fixed RNG seeds; save artefacts under `.benchmarks/`.
- Assertions: use `pytest.approx`, `torch.testing.assert_close`, or `math.isclose` appropriately.

## 7) Performance (facts)

- Start with pure Python/Torch; introduce C++ for clear hotspots only.
- Benchmarks live in `tests/performance/`; use `just bench` to run smoke benches.
- Profile on demand with local tools; notebooks are fine for exploration.

## 8) Workflows (imperative)

Daily development

1. Read task; scan relevant code/tests.
2. Write a short plan (4–7 steps) and implement minimal, cohesive changes.
3. Keep math pure; do I/O only in datasets/models/adapters.
4. Run `just precommit` before handoff (format, lint, type, smoke tests).

Pre‑PR checks

- Keep diffs focused; update tests/docs accordingly.
- Ensure `just ci` is green locally.

PR content

- Scope, files touched, what changed, how tested (Ruff/Pyright/pytest summaries), perf delta if relevant, limitations and follow‑ups.

Blocked?

- After ~60–90 minutes blocked, open a draft PR with `Needs-Unblock: <topic>`.

## 9) CI & Branch Protection (facts)

- `just ci` mirrors GitHub Actions: lint, type, and smoke tests; weekly jobs may run longer benches later.

## 10) Policy Waivers (facts)

- Deviations live in `waivers.toml` with `id`, `summary`, `owner`, `scope`, `created`, `expires`, `justification`, `removal_plan`.

## 11) Module Layout & Conventions (facts)

- Namespaces
  - `viterbo.math`: pure geometry/math utilities (Torch tensors I/O).
  - `viterbo.datasets`: adapters/datasets/collate functions for ragged data; thin wrappers around math.
  - `viterbo.models`: experiments/training loops; may use GPU; no core math here.
  - `viterbo._cpp`: C++/pybind11 extensions (CPU baseline).

- Conventions
  - Use semantic names like `normals`, `offsets` for halfspaces `Bx ≤ c` where applicable.
  - Avoid dataclasses in `math`; return tensors/tuples of tensors.
  - Keep `datasets` simple; no DSL; explicit functions and small classes.

## 12) Clarifications & Bans (imperative)

- No `__all__` across the repository.
- Avoid custom typedefs for shapes/dimensions; annotate directly at call sites.
- Maintain strict layering: `math` must not depend on `datasets`/`models`.
- Prefer small, focused cleanups over exceptions. Use waivers only when necessary and time‑boxed.
