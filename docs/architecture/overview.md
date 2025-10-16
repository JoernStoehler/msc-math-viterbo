# Architecture Overview (Reference)

This document records decisions and rationale that go beyond the everyday overview in `AGENTS.md`.
It is intended for maintainers or agents working on architecture/conventions.

- Stack & devices
  - PyTorch-first; CPU baseline across CI and default runs.
  - GPU is optional and used only in `models/` where it improves throughput for conventional ML.
  - Math functions accept the caller’s device; no implicit device moves.

- Dtypes & numerics
  - Dtype expectations are documented per function/docstring (math often float64; ML often float32).
  - Avoid silent downcasts. Prefer numerically stable formulations and document trade-offs.

- Layering (no cycles)
  - `viterbo.math`: pure, stateless Torch tensor functions; no I/O; no dataclasses.
  - `viterbo.datasets`: thin adapters, schemas, and ragged collators; may cache or precompute.
  - `viterbo.models`: experiments and training loops; no core math logic.
  - `_cpp`: C++ extensions via `torch.utils.cpp_extension` (CPU baseline), with Python fallbacks.

- Ragged data patterns
  - Use Python lists of tensors or padded tensors + masks depending on the consumer.
  - Collate functions live in `viterbo.datasets` (e.g., `collate_list`, `collate_pad`).

- C++ extensions
  - Baseline is CPU-only C++ compiled via `torch.utils.cpp_extension.load` with pybind11 bindings.
  - Keep a safe Python fallback for each extension to preserve portability (CI, devcontainers).
  - Add CUDA only when a clear hotspot is demonstrated by profiling and justified by complexity.
  - Example harness lives in `src/viterbo/_cpp/`: single-file (`add_one.cpp`) and multi-file (`affine_ops.{h,cpp}` + `affine_bindings.cpp`) builds are both wired through lazy loaders.
  - Local build trigger (devcontainer validated): `uv run python -c "import torch; import viterbo._cpp as cpp; cpp.add_one(torch.ones(1)); cpp.affine_scale_shift(torch.ones(1), 1.5, 0.5)"`. This compiles the extensions once per Python process, caching objects under `.cache/torch_extensions/`.
  - Default flags include `-O3`; pass more via `TORCH_CUDA_ARCH_LIST`, `CC`, `CXX`, or `CFLAGS` environment variables before import when platform tuning is required.
  - Common failures: missing compilers (`c++: not found` → install `build-essential`), stale build artefacts (`rm -rf ~/.cache/torch_extensions/*`), or Ninja absence (`uv sync` pulls the bundled `ninja` dependency, otherwise `uv add ninja`); fallbacks keep runtime functional meanwhile.

- Testing & CI philosophy
  - Smoke-first: quick validators and selective benchmarks in PRs; deeper tiers are opt-in.
  - Incremental selection for fast feedback loops; CI uses CPU-only Torch wheels for speed.
  - CI exports `PIP_INDEX_URL`/`UV_TORCH_BACKEND=cpu` so installers resolve against the CPU-only PyTorch index; developers override locally when they need CUDA wheels.
  - Docs are built in CI so documentation drift is surfaced alongside code changes.

- Imports & public surface
  - Prefer explicit imports from concrete modules; do not aggregate or re-export APIs in `__init__.py`.
  - No `__all__` surfaces. Import paths should remain stable by direct module references.

- Tasks & parallelization
- Author small, self-contained tickets directly in the VibeKanban board (clear scope, ACs) and link to supporting briefs in `docs/` as needed.
  - Keep branches short-lived; ensure CI green before review; record any deviations.
