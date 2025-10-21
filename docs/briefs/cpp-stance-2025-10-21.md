Title: C++ Extensions Stance — Strict Tooling, No Fallbacks
Status: Draft (Owner Decision Required)
Owners: Jörn Stöhler (Owner); Advisors: Kai Cieliebak (optional)
Date: 2025-10-21

Context / Problem
- Tests cover a minimal C++ extension surface: `tests/test_cpp_extension.py` and an optional benchmark: `tests/performance/test_cpp_extension_benchmark.py`.
- The architecture (docs/architecture/overview.md) describes CPU-only Torch C++ extensions. Prior drafts allowed Python fallbacks on build errors; the Owner requested removal of masking behavior.
- Goals (explicit):
  - Ensure C++ tooling works reliably in devcontainer/CI.
  - Do not hide when C++ tooling fails.
  - Do not hide when C++ code fails (static/runtime bugs).
  - Python implementations are acceptable as separate references for testing, but not as runtime fallbacks.

Proposals

Option C — Enforce C++ strictly in core (no fallbacks)
- What
  - Make extension loaders strict: on build/import failure, raise a dedicated `CppExtensionUnavailableError` with diagnostics. No automatic Python fallback in production code paths.
  - Keep separate Python reference implementations under a distinct namespace (e.g., `viterbo._ref`), used only in tests/benchmarks for cross‑checking.
- Pros
  - Surfaces toolchain and C++ failures immediately; no masking.
  - Preserves ability to test correctness via `_ref` without conflation.
- Cons
  - Environments without compilers will fail fast; requires explicit opt‑out tracks for non‑C++ development (see D).
- CI/devcontainer implications
  - Devcontainer: must have `build-essential`, `ninja`. First import compiles and caches; failures fail tests.
  - CI: add a prebuild step to fail early; enable verbose logs.
- Minimal diff sketch (paths only; no code)
  - `src/viterbo/_cpp/__init__.py` — raise on build/import failure; drop silent excepts and fallbacks.
  - `src/viterbo/_ref/__init__.py` — add Python reference ops (e.g., `add_one_ref`, `affine_scale_shift_ref`).
  - `tests/test_cpp_extension.py` — remove fallback test; parametrize comparisons `_cpp` vs `_ref`.
  - `docs/architecture/overview.md` — update to “no fallbacks in core; references live under `_ref` for tests”.

Option D — Strict by default, with explicit non‑C++ test track
- What
  - Default `just test` and CI require C++ to pass.
  - Provide a developer‑only command to deselect C++ tests (e.g., `just test-no-cpp` → `-m "not cpp"`); this is visible and explicit, not a fallback.
- Pros
  - Keeps strictness where it matters (CI, mainline dev) while unblocking contributors lacking compilers temporarily.
  - Visibility preserved: the C++ test set is excluded, not faked.
- Cons
  - Requires marker plumbing and a separate just target; slightly more surface.
- Minimal diff sketch (paths only; no code)
  - `tests/test_cpp_extension.py` — add `@pytest.mark.cpp` to C++ tests.
  - `tests/performance/test_cpp_extension_benchmark.py` — same `@pytest.mark.cpp` (and keep `pytest_benchmark` guard).
  - `Justfile` — add `test-no-cpp` target selecting `-m "not cpp"`.
  - `README.md` — document strict default and optional developer track.

Option E — Prebuild gate + diagnostic failure
- What
  - Add a preflight compile step that builds a tiny extension once and surfaces environment issues early (compiler path, Ninja presence, include dirs).
  - Use `VITERBO_CPP_VERBOSE=1` in CI and emit a short failure guide.
- Pros
  - Fails fast with crisp logs; avoids spending test time only to fail later.
- Cons
  - Slight CI latency; additional script maintenance.
- Minimal diff sketch (paths only; no code)
  - `.github/workflows/ci.yml` — add “C++ prebuild” step (before tests).
  - `scripts/cpp_preflight.py` — compile a trivial op via `torch.utils.cpp_extension.load`.
  - `docs/architecture/overview.md` — note preflight behavior.

Option F — Tooling hardening & policy
- What
  - Declare Ninja default (`USE_NINJA=1`), keep `ninja` in `[project.dependencies]`.
  - Document required toolchain in devcontainer and local setup; provide quick troubleshooting (cache path, `MAX_JOBS`, verbose).
- Pros
  - Reduces flakes, aligns expectations; easy for new contributors.
- Minimal diff sketch (paths only; no code)
  - `README.md` — C++ toolchain requirements + troubleshooting.
  - `docs/architecture/overview.md` — explicit “no fallback” policy.

Analysis & Trade‑offs
- C strictly enforces correctness and visibility; D offers a deliberate, explicit escape hatch for local dev without masking; E/F improve reliability and diagnostics.
- None of these options rely on Python fallbacks in production pathways.

Recommendation
- Primary: Option C (strict, no fallbacks in core) + Option F (tooling hardening).
- Optional: Add Option D for developer convenience only; CI stays strict. Option E is recommended if we want even earlier failure and clearer logs.

Risks & Mitigations
- Risk: Contributors without compilers blocked on default tests.
  - Mitigation: Provide `test-no-cpp` target (D) for local only; CI remains strict.
- Risk: CI failures due to environment changes (e.g., missing Ninja).
  - Mitigation: Pin `ninja`, add prebuild gate (E), and document recovery steps.

Rollback Plan
- If strictness causes undue friction, we can temporarily use D to keep local loops moving while investigating, without introducing runtime fallbacks. Revert by re‑enabling cpp marker selection in defaults.

Owner Decision Block (copy/paste)
- [ ] Choose stance: Primary — C (strict, no fallbacks) / D (strict-by-default + dev-only no-cpp track)
- [ ] Optional add‑ons: E (prebuild gate) / F (tooling hardening in docs)
- [ ] Confirm Ninja policy (`USE_NINJA=1`) and keeping `ninja` in deps
- [ ] Confirm test marker plan (`@pytest.mark.cpp`) and removal of fallback tests
- [ ] Approve follow‑up implementation tasks to apply selected options

References
- Tests: `tests/test_cpp_extension.py`, `tests/performance/test_cpp_extension_benchmark.py`
- Docs: `docs/architecture/overview.md`, `docs/reviews/15-project-complexity.md`
- Code: `src/viterbo/_cpp/`

Provenance
- Prepared by: Codex CLI Agent (docs‑only change in this task)
