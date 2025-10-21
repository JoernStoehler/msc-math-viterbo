# Review 15 — Project Complexity

Provenance
- Topic: 15 — Project complexity (agent onboarding and day‑to‑day work)
- Author: Codex CLI Agent
- Date: 2025-10-21
- Commit: 14352dd
- Timebox: ~90 minutes
- Scope: Assess features/aspects that increase or reduce complexity for Codex agents during normal tasks (coding, documenting, testing, reading papers, thesis writing, VK planning). Focus on reality in this worktree.
- Version: v0.1

Context
- Goal: Identify concrete sources of complexity that overwhelm new agents and slow routine work, and call out aspects that are neutral or helpful (low complexity). Prioritise actionable observations over theory.

Inputs & Method
- Ran: `uv run python scripts/load_skills_metadata.py --quiet` — to surface skills context (AGENTS.md integration) (scripts/load_skills_metadata.py:1)
- Scanned repo with `rg --files` and opened representative files (≤250 lines per read as per policy).
- Read: `AGENTS.md`, `Justfile`, `pyproject.toml`, `sitecustomize.py`, `mkdocs.yml`, `README.md`, selected `docs/math/*.md`, selected `docs/reviews/*.md`, and tests under `tests/**`.
- Checked current code under `src/viterbo/**` and compared against tests/docs to assess mismatches.

Findings (unsorted)
- Project breadth (docs, code, tests, notebooks, thesis, devcontainer, Cloudflare workers) forces context switching; this breadth is a complexity driver for onboarding.
- Skills meta‑system (AGENTS.md + skills/*.md) is helpful but adds a procedural layer; agents must learn and remember to run `uv run python scripts/load_skills_metadata.py` early (AGENTS.md:12).
- Golden commands via `Justfile` reduce guesswork but introduce their own vocabulary (`preflight`, incremental selector, smoke/deep/longhaul tiers) (Justfile:1).
- Incremental test selector (`scripts/inc_select.py`) is powerful yet conceptually heavy (import graph, hashing, thresholds, plumbing fallbacks). New agents need to internalize “why tests didn’t run” when it exits with “skip” (scripts/inc_select.py:1).
- CPU time cap in `sitecustomize.py` (default 180s) can surprise agents running long local tasks; not knowing about `VITERBO_CPU_LIMIT=0` leads to abrupt terminations (sitecustomize.py:27).
- Mixed programming surfaces (Python/Torch, C++ extensions, LaTeX thesis, Cloudflare workers) increase cognitive load even if not all are used on a given task (.devcontainer/cloudflare/*, thesis/*).
- Devcontainer provisioning includes multiple host mounts, Cloudflare tunnel env, and VS Code extensions; this helps maintainers but is a mental burden for agents not using containers (.devcontainer/devcontainer.json:1).
- README advertises devcontainer variants that aren’t present (“devcontainer.local.json”, “devcontainer.codespaces.json”); mismatch erodes trust (README.md:36).
- Heavy test suite references to modules that do not exist in this worktree (e.g., `viterbo.math.*`, `viterbo._cpp`) cause immediate failure or confusion; agents must decide whether to implement, skip, or adjust tests (tests/math/test_polytope.py:1; tests/test_cpp_extension.py:1), while src contains only `src/viterbo/runtime.py` and `src/viterbo/__init__.py`.
- Docs claim implementations that are not present (e.g., “Oriented‑Edge CH Algorithm — Status: Implemented”) which contradicts the code tree; this is a major complexity/ambiguity source (docs/math/oriented_edge_spectrum_4d.md:1 vs src/viterbo/**).
- Reviews and combined review pages assume modules that aren’t in this worktree (e.g., polytope/volume/capacity stacks); the narrative is high‑quality but misaligned with reality, leading to “fight the docs” friction (docs/reviews/03-repo-architecture.md:1; docs/reviews/combined.md:1).
- Tests under `tests/datasets/` reference `viterbo.datasets.atlas_tiny` APIs not present under `src/`; this increases triage complexity for any dataset‑related task (tests/datasets/test_atlas_tiny.py:1).
- PyTorch pinned to ≥2.5 with NumPy ≥2.3.3 and SciPy ≥1.15.2 implies a modern stack; helps consistency but increases setup friction for agents expecting older PyTorch or pure‑NumPy flows (pyproject.toml:9).
- `ninja` is included as a dependency (pyproject.toml:19) and tests expect C++ extensions (`viterbo._cpp`), but no `_cpp/` sources exist; agents may waste time hunting for missing code (tests/test_cpp_extension.py:1).
- Strict lint/type policies (Ruff + Pyright) create initial friction but are net helpful once learned; the Justfile wraps them, which is good (pyproject.toml:43; Justfile:138).
- Lint rules include pydocstyle with Google convention; good for consistency but adds verbosity and reformat churn for newcomers (pyproject.toml:63).
- Import policy (“no re‑export indirection”) is stated (src/viterbo/__init__.py:1) but much of the referenced surface is missing; agents can’t rely on import stability yet.
- The reviews index template expects checks like `uv run mkdocs build --strict`; if the code/docs mismatch causes build failures, agents face non‑obvious fixes (docs/reviews/README.md:1; mkdocs.yml:1).
- `mkdocs.yml` advertises a Math API nav (polytope, constructions, volume, symplectic, capacity_ehz, oriented_edge, etc.) without corresponding code; creates pressure to implement multiple modules at once (mkdocs.yml:15).
- The “Always‑On Essentials” (AGENTS.md) help but are dense; novice agents may skim and miss important details like “stream reads ≤250 lines” or “do not revert files you didn’t edit” (AGENTS.md:17).
- Multiple review documents and deep‑dives are excellent context for maintainers but overwhelm newcomers; it’s hard to know which statements are actionable today vs aspirational (docs/reviews/*.md).
- The repository mixes stable, immutable content (papers under docs/papers/*) with fast‑moving guidance; agents must learn what to treat as reference vs mutable plan — not always marked clearly.
- The `notebooks/` directory includes small helper runners; expectations around Jupytext and artefact hygiene exist in skills docs, which is helpful, but add yet another mental model (notebooks/*.py; skills/experiments-notebooks-artefacts.md:1).
- The “Atlas Tiny” docs exist (docs/datasets/atlas_tiny.md:1) but the dataset code is absent; gap introduces friction when tests refer to it.
- Preflight gate in Justfile enforces worktree branch naming (`vk/<worktree-name>`) and prevents detached HEAD; helpful but can block first‑time runners until they understand the pattern (Justfile:23).
- The incremental selector stores state under `.cache/`; agents may not notice when prior failures are re‑selected automatically; invisible state increases perceived randomness (scripts/inc_select.py:46).
- CI docs build is strict; minor markdown or nav drifts can fail local CI; helpful quality bar, but yet another moving part to consider on doc changes (mkdocs.yml:1; README.md:1 badges).
- `sitecustomize.py` also tweaks JAX config despite JAX not being a declared dependency; this “surprising import” may trigger warnings in some environments (sitecustomize.py:57).
- Skills metadata frontmatter is manually curated and validated; frontmatter mistakes fail `just lint` and add to setup overhead when only touching docs (scripts/load_skills_metadata.py:1).
- The “VK‑safe formatting” rules (skills/vibekanban.md) add formatting constraints on ticket texts; helpful in VK, but an extra set of rules to remember in addition to markdown norms (skills/vibekanban.md:1).
- Code style emphasizes explicit dtypes/devices and forbids implicit moves; very good for math quality, but sometimes verbose and unintuitive for non‑Torch users (skills/math-layer.md:1; skills/good-code-loop.md:1).
- The tests rely on deterministic Torch RNG usage via `torch.Generator`; learning this pattern is easy but another concept for newcomers (tests/polytopes.py:1).
- Several tests are parametrised and combine geometry and symplectic invariants; great for coverage but daunting to debug without the underlying implementations (tests/math/test_capacity_ehz_haim_kislev.py:1).
- Performance tests/benchmarks exist and expect optional plugins; running them without `pytest-benchmark` skips with markers, which is fine but might confuse expectations (tests/performance/test_cpp_extension_benchmark.py:1).
- The project encourages “Torch‑first” numerics and float64 for math; good policy but occasionally clashes with default float32 habits, requiring explicit upcasts in new code (skills/math-layer.md:23).
- The `docs/reviews/combined.md` lists “Decisions Required” that imply prior actions (e.g., secrets remediation, coverage floors) that are not clearly reflected in current configs; agents need explicit status (docs/reviews/combined.md:1).
- Tests mention “oriented‑edge memo/budgets” controls and invariances; without code, agents can’t validate or learn from them, turning tests into blockers rather than guides (tests/math/test_capacity_oriented_edge.py:1; tests/math/test_oriented_edge_memo_smoke.py:1).
- The mail/ folder introduces process templates and archives; useful for the owner, but largely irrelevant to agents and adds to noise (mail/*.md).
- `Justfile` uses environment variables (`UV_PROJECT_ENVIRONMENT`, `BENCHMARK_STORAGE`, etc.) and mounts in devcontainer; magic paths can surprise when running outside containers (Justfile:7).
- Some reviews suggest coverage floors and CI gates, but `pyproject.toml` does not set `report:fail_under`; agents can’t tell if failing coverage is expected to block PRs (pyproject.toml:95; docs/reviews/combined.md:1).
- `pytest.ini` exists but is minimal; the real selection is custom; this divergence from “pytest‑as‑usual” norms adds learning curve (pytest.ini:1; scripts/inc_select.py:1).
- The `AGENTS.md` skill map is long; while it points to the right places, finding the minimum set to read per task is non‑trivial (AGENTS.md:1).
- The repo requires `uv` instead of pip/venv; `uv` is great but non‑standard for many; initial friction is common (README.md:18; Justfile:62).
- Docs/Math pages include formalism (symplectic forms, budgets, CH algorithm); for feature tasks, this is reference material, but during onboarding it can feel like mandatory study even for non‑math tasks (docs/math/*.md).
- The `docs/reviews/` process is structured with checklists; excellent for rigor, but the friction of filling Provenance/Validation may feel heavy for small reviews (docs/reviews/TEMPLATE.md:1).
- The tests use `torch.set_default_dtype(torch.float64)` in many files; agents must be careful not to unintentionally rely on default dtype elsewhere (tests/math/test_polytope.py:9).
- Many tests include `@pytest.mark.skip` placeholders for planned backends; this is good to communicate intent, but the volume of planned paths adds perceived scope (tests/math/test_polytope.py:29).
- Devices: GPU is optional and not well surfaced; agents may try to run on CUDA inadvertently; lack of documented CUDA policy in this worktree simplifies for now but leaves ambiguity later (docs/architecture/overview.md:5).
- The `docs/papers/` directory mixes Markdown and LaTeX; file formats and references vary; not inherently hard, but navigation requires discipline (docs/papers/*).
- “Atlas Tiny” tests imply dataset collators, masks, and schemas; without code, it’s ambiguous whether to implement full pipeline or stub (tests/datasets/test_atlas_tiny.py:1).
- The “per‑call wall‑clock budget” decorator exists and is documented (src/viterbo/runtime.py:1); a guard that adds predictability, not complexity.
- Some doc pages include status banners (“Implemented”) that may be inaccurate; status hygiene is a big source of confusion that compounds complexity (docs/math/oriented_edge_spectrum_4d.md:1).
- The repo has strong language and formatting norms (Unicode math, VK renderer caveats); split rules (code vs VK vs docs) means agents must recall three surfaces of formatting (skills/vibekanban.md:33).
- The `Justfile` “preflight” prints helpful diagnostics (branch, rebase state, ahead/behind vs main); visibility reduces complexity once discovered (Justfile:23).
- The skill “good‑code‑loop” sets clear layering rules and coding standards; decreases complexity by reducing choices (skills/good-code-loop.md:1).
- The skill “math‑layer” is explicit about purity and vectorization; good for implementation discipline; net complexity reducer for math tasks (skills/math-layer.md:1).
- The presence of rich tests for polytope conversions, volume, capacity, and invariants is a map for implementation, but only when the referenced module structure exists; in this worktree, it becomes a source of confusion (tests/math/*.py).
- The `typings/` directory contains JAX/type stubs unused elsewhere; stubs without code references can add noise (typings/jax/*.pyi).
- GitHub workflows for docs/CI exist; helpful for parity but add mental overhead for agents who only needed to write code for a single test (/.github/workflows/*.yml).
- Benchmarks default to autosave under `.benchmarks`; leaving such artefacts out of Git is correct, but the path tokens (`{{BENCHMARK_STORAGE}}`) appearing in rg output can confuse (tests/performance/*; Justfile:32).
- Reviews/meeting pages are numerous; the “Run of Show” is useful for meetings but not directly for implementation tasks (docs/meeting/*.md).
- The thesis directory (LaTeX) is a separate toolchain; agents working on code shouldn’t need to touch it, but its presence can suggest extra duties (thesis/*).
- Code quarantines (don’t revert files you didn’t edit) and explicit escalation tags (`Needs‑Unblock:`) reduce accidental thrash; net positive (AGENTS.md:19).
- The repo enforces short line length (100) and import ordering; simple but causes reformat churn before habits form (pyproject.toml:36).
- The `README.md` quickstart is concise and good; helps reduce complexity when followed (README.md:16).
- The docs nav exposes many math pages; useful index but invites deep reading; consider marking priority reads (mkdocs.yml:15).
- The mismatch between “advertised” repository layout (README) and actual code state (src minimal) is the single largest complexity multiplier for agents right now.
- The combination of ambitious math scope and C++ extension plans with scant code stubs makes task boundaries unclear; agents need a “current milestone” scope to avoid boiling the ocean.
- The project relies on task planning via VibeKanban; the skill provides good rules, decreasing complexity once a rhythm is established (skills/vibekanban.md:1).
- The presence of many prior reviews can be leveraged to cut ambiguity, but only if each review’s statements are versioned or tagged with “applies/does not apply to current worktree”.
- Some tests assert invariance under translation/scaling (capacity oriented‑edge); great practice, but without code is another blocker (tests/math/test_capacity_oriented_edge.py:1).
- The `scripts/fetch_paper.py` and papers workflow is a nice accelerator for literature; for pure coding tasks this is noise; for reading tasks, it reduces complexity (scripts/fetch_paper.py:1; skills/papers-workflow.md:1).

Actions (first pass)
- Align docs with reality: add a status banner to each docs/math page and review stating “Implemented / Planned / Missing in this worktree”, and fix obvious mismatches (e.g., oriented‑edge page claims implemented) — owners to approve wording.
- Add a “start here” landing page for agents with two tracks: (A) code task track (minimum skills to read; 10‑minute loop) and (B) docs/lit track; link from README and AGENTS.md.
- Create minimal stubs for referenced top‑level modules (`src/viterbo/math/{polytope,constructions,volume}/__init__.py` and a couple of tiny functions) and mark tests that require deeper impl as `xfail` with clear reason; reduces red wall and guides incremental delivery.
- Decide on C++ extension scaffolding: either add `_cpp` minimal loaders with Python fallback to satisfy smoke tests or skip the C++ tests behind a marker until sources land; update pyproject/README accordingly.
- Gate the incremental selector behind a `JUST_USE_SELECTOR=1` toggle for newcomers; default to full smoke runs in `just test` while keeping `just test-fast` for selector users.
- Surface `VITERBO_CPU_LIMIT` prominently in README quickstart and `just help`; add a `just longrun` example with the var disabled for approved long tasks.
- Add a “Current Milestone” doc stating exactly which math APIs are in scope this iteration and which tests are authoritative; link from VK tickets.

Cross-References
- Files: `AGENTS.md:1`, `Justfile:1`, `pyproject.toml:1`, `sitecustomize.py:1`, `mkdocs.yml:1`, `README.md:1`, `docs/math/oriented_edge_spectrum_4d.md:1`, `tests/math/test_polytope.py:1`, `tests/datasets/test_atlas_tiny.py:1`, `tests/test_cpp_extension.py:1`, `scripts/inc_select.py:1`.
- Topics: T02 (Onboarding), T03 (Repo architecture), T05 (Developer docs), Deep‑Testing & CI, Deep‑EHZ 4D, Deep‑CPP toolchain.

Validation
- `uv run python scripts/load_skills_metadata.py --quiet` — OK (skills listed; AGENTS.md sections present).
- `rg --files` — inspected tree; verified code/docs/test mismatches.
- No `just` or pytest runs executed in this pass to avoid broad changes; focus was structural review.

Limitations
- Did not execute CI or docs build; some mkdocs claims not validated end‑to‑end.
- Did not measure performance; observations about benchmarks are structural only.
- Reviewed representative files; some nuances may exist in un‑opened files.

Status
- Draft

Pre-Submit Checklist
- Linked from `docs/reviews/README.md` — Pending (owner can add link if desired).
- `uv run mkdocs build --strict` — Not run in this pass.
- `just checks` — Not run in this pass (docs‑only change).
- Actions extracted — Yes (first pass above).
- Cross‑refs noted — Yes.
- Provenance filled — Yes.
- Title matches pattern — Yes.

