# Review — C++ Extensions & Build Toolchain Deep-Dive

Provenance
- Topic: C++ Extensions & Build Toolchain Deep-Dive
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 87fdebc
- Timebox: 60–90 minutes
- Scope: extension shim/fallback policy, build reproducibility (torch.utils.cpp_extension), safe exceptions, Python fallbacks, local build ergonomics, and CI behavior. Excludes CUDA/GPU builds and cross-platform Windows specifics.
- Version: v0.1

Context
- Goal: assess how C++ extensions are integrated, how robust the shim/fallback is, whether local + CI builds remain reproducible/ergonomic, and what failure modes exist.

Inputs & Method
- Code scan: `rg -n "torch.utils.cpp_extension|cpp_extension|load\(" -S`
- Read: `src/viterbo/_cpp/__init__.py`, `src/viterbo/_cpp/*.cpp|*.h`, tests around extensions.
- Docs: `docs/architecture/overview.md`, prior review pages mentioning extensions.
- Skills index refresh (non-mutating): `uv run python scripts/load_skills_metadata.py --quiet`
- Docs build (strict): `uv run mkdocs build --strict`

Findings (unsorted)
- Shim location and shape
  - C++ entry lives at `src/viterbo/_cpp/__init__.py:1` with lazy loaders for each op.
  - `from torch.utils.cpp_extension import load` at import time implies Torch must import to reach fallbacks; acceptable given project dependency pin `torch>=2.5.1,<2.6` (pyproject.toml:10).
  - Extension loaders are `@lru_cache`d to avoid repeated compiles per process (`src/viterbo/_cpp/__init__.py:40,56`).
  - Fallbacks exist for each op: `add_one` and `affine_scale_shift` dispatch to pure Torch when extension is unavailable.

- Exception policy
  - `_SAFE_EXCEPTIONS = (OSError, RuntimeError, ImportError)` (`src/viterbo/_cpp/__init__.py:17`). These cover many common build/import errors but may miss `subprocess.CalledProcessError` or toolchain-specific exceptions (e.g., `distutils.errors.CompileError` in some environments).
  - Current behavior: swallow safe errors and return `None` module → Python fallback path. Good for portability; can hinder diagnosis unless logs are visible elsewhere.

- Build driver and flags
  - Loader sets `USE_NINJA=0` by default (`src/viterbo/_cpp/__init__.py:28`). Comment mentions Torch<=2.4; project pins Torch 2.5.1. Default-off Ninja is conservative/portable but contrasts with a declared dependency on `ninja>=1.13.0` (pyproject.toml:16).
  - `extra_cflags=["-O3"]` only; no explicit `-march`, `-mtune`, or determinism flags (e.g., `-fno-ident`, `-Wl,--build-id=none`). Reproducibility is pragmatic (behavioral), not byte-for-byte.
  - `verbose=False` suppresses build logs. In failure scenarios this keeps tests/dev loops quiet but obscures root causes.

- Reproducibility posture (torch.utils.cpp_extension)
  - Uses dynamic `load(...)` which hashes sources/flags into a path under `~/.cache/torch_extensions` (Torch behavior). Rebuilds when inputs change; otherwise caches binaries.
  - No project-local `build_directory` override; global cache can mask environment diffs across branches/users unless explicitly cleaned.
  - Compiler selection not pinned; relies on system `c++`. Determinism varies by container/host.

- Python fallbacks
  - `add_one`: fallback `x + 1` (`src/viterbo/_cpp/__init__.py:53`) matches C++ semantics.
  - `affine_scale_shift`: casts `scale/shift` to float for C++ path (`:75`) and uses `x * scale + shift` fallback (`:76`). Behavior consistent; dtype/device preserved by Torch.
  - Tests cover both correctness and forced fallback via monkeypatch (`tests/test_cpp_extension.py:19-38`).

- Safety and API surface
  - Public capability checks: `has_add_one_extension()` and `has_affine_extension()` allow callers/tests to branch behaviorually (`src/viterbo/_cpp/__init__.py:79-86`).
  - Cache reset helper `clear_extension_caches()` aids tests (`:89-92`).
  - No dtype/device assumptions; operations broadcast on tensors as expected.

- Local developer ergonomics
  - Dev flow compiles lazily on first call; example commands are documented in `docs/architecture/overview.md` (see lines referencing lazy compile and cache).
  - Absent helper to purge Torch extension cache (`~/.cache/torch_extensions`), though local `clear_extension_caches()` clears the in-process memoization.
  - Silence by default (`verbose=False`) makes day-one friction low; when compilers are missing, fallbacks mask problems—useful for portability, but can hide performance regressions.
  - Pin to Torch CPU wheel in CI plus local `uv` flows keeps environment consistent. Developer machines with CUDA won’t accidentally flip CI behavior.

- CI behavior
  - `just ci` installs dev deps, runs lint/type/smoke tests, and builds docs strictly (`Justfile: ci`). Smoke tests include `tests/test_cpp_extension.py`, which pass even if builds fail (fallback validated).
  - `just ci-cpu` installs CPU-only Torch into system site and runs tests/docs without `uv run` for runtime steps. Good separation. C++ extension compilation is optional; tests do not require success to be green.
  - Benchmarks skip when the extension is unavailable (`tests/performance/test_cpp_extension_benchmark.py`), preventing noisy failures.

- Mismatch: Ninja dependency vs default-off Ninja
  - `pyproject.toml` declares `ninja>=1.13.0`, but loader forcibly disables Ninja by `USE_NINJA=0` unless overridden by the environment. This is internally inconsistent. Options: (a) remove `ninja` dependency, or (b) default to Ninja when present and allow `USE_NINJA=0` to disable.

- Comment drift
  - Comment says “Torch<=2.4 expects Ninja by default” (`src/viterbo/_cpp/__init__.py:25`). Torch 2.5 still prefers Ninja when available; wording should be updated to avoid version-specific ambiguity.

- Failure modes observed/anticipated
  - Missing compiler (`c++: not found`) → caught as `OSError`/`RuntimeError`; fallback taken; no logs due to `verbose=False`.
  - Incompatible flags/toolchain warnings won’t fail builds; may change performance subtly.
  - Stale global cache can cause confusing behavior when switching branches or flags; not project-scoped.
  - Import-time hard dependency on Torch (module imports Torch before attempting extension load); if Torch missing, import fails (acceptable as project depends on Torch).

- Policy fit with architecture docs
  - Architecture page states CPU-first C++ via `torch.utils.cpp_extension.load` with fallbacks and advises local command that triggers builds; aligned with code.
  - Docs mention `-O3` and typical failure remedies; consistent with implementation.

- Test coverage
  - Smoke tests validate API correctness and fallback forcing.
  - No explicit assertion that the extension actually compiled in CI; intentionally avoided to keep CI green on environments without compilers.
  - No negative tests around non-contiguous tensors or dtypes; optional future additions if ops become non-trivial.

- Ergonomics gaps and quick wins
  - No `just` target to clean the global Torch extension cache or to force a one-shot local compile smoke run.
  - No env-driven verbosity gate for extension build logs (e.g., `VITERBO_CPP_VERBOSE=1`).
  - Exception tuple not covering `CalledProcessError`/`CompileError` may surface tracebacks on some platforms instead of falling back.

Actions (first pass)
- Align Ninja policy and dependency
  - Decide one: remove `ninja` from project deps and keep `USE_NINJA=0` default, or honor Ninja by default when present (set `os.environ.setdefault("USE_NINJA", os.getenv("USE_NINJA", "1"))`) and keep doc guidance. Update comment accordingly.
- Expand safe exceptions
  - Add `subprocess.CalledProcessError` and (if available) `distutils.errors.CompileError` to `_SAFE_EXCEPTIONS` so more build failures hit clean fallbacks.
- Add opt-in build verbosity
  - Gate `verbose` via `VITERBO_CPP_VERBOSE` env var and optionally echo a short one-line warning when falling back (behind `VITERBO_CPP_WARN_FALLBACK=1`).
- Add local ergonomics helpers
  - New `just` targets: `ext-clean` to remove `~/.cache/torch_extensions/*`, and `ext-smoke` to run a tiny compile/exercise snippet. Document in `docs/architecture/overview.md`.
- Consider project-local build directory
  - Pass `build_directory=str(Path(".cache/torch_extensions_local").absolute())` to `load(...)` for easier cleanup and per-repo isolation (optional).
- Optional CI sentinel job (non-blocking)
  - Add an opt-in CI job that compiles the extension once (Linux, CPU) and reports time/logs without failing the main pipeline when the compiler is absent. Keep smoke tier unchanged.

Cross-References
- Topics: Architecture Overview, Code Review
- Files:
  - `src/viterbo/_cpp/__init__.py:14`
  - `src/viterbo/_cpp/__init__.py:24`
  - `src/viterbo/_cpp/__init__.py:28-37`
  - `src/viterbo/_cpp/add_one.cpp:1`
  - `src/viterbo/_cpp/affine_ops.cpp:1`
  - `src/viterbo/_cpp/affine_bindings.cpp:1`
  - `tests/test_cpp_extension.py:1`
  - `tests/performance/test_cpp_extension_benchmark.py:1`
  - `docs/architecture/overview.md:19`
  - `pyproject.toml:1`
  - `Justfile:1`

Validation
- `uv run python scripts/load_skills_metadata.py --quiet` — OK (no output; non-mutating)
- `uv run mkdocs build --strict` — OK; build completed; INFO about pages not in nav is expected and not a warning.
- Repo scan: `rg -n "torch.utils.cpp_extension|cpp_extension|load\(" -S` — surfaced loader uses and relevant docs/reviews.

Limitations
- Did not attempt GPU/CUDA extension builds; CPU-only focus.
- Did not verify byte-for-byte deterministic builds; focused on behavioral reproducibility and ergonomics.
- Did not run full test matrix; smoke-only assumptions based on `Justfile` and test content.
- Windows toolchain behavior not reviewed.

Status
- Draft

Pre-Submit Checklist
- Link from `docs/reviews/README.md` — deferred by guardrails (no index edits).
- `uv run mkdocs build --strict` — green (recorded).
- `just checks` — N/A for docs-only change; not run to avoid unrelated edits.
- Actions extracted — present (first pass).
- Cross-refs noted — present.
- Provenance — filled.
- Title — follows review pattern (unnumbered per task guardrails).

