Status: Implemented (scope: project design principles review snapshot; caveats: reflects repository state on 2025-10-20)

# Review 01 — Project Design Principles (Onboarding Focus)

Provenance
- Topic: 01 — Project Design Principles
- Author: Codex CLI Agent
- Date: 2025-10-20
- Commit: 96e7f5a
- Timebox: ~60–90 minutes
- Scope: Design principles with onboarding emphasis
- Version: v0.1

Context
- Assess whether repository-level principles enable the 6‑month MSc thesis goals, with emphasis on onboarding speed, readability, test posture, and math correctness.

Inputs & Method
- Skimmed README, skills, Justfile, CI/test setup, and math docs; spot-checked test coverage areas.

Findings (unsorted)
Unfiltered, unsorted list of observations related to project-level design principles that enable a successful 6‑month MSc thesis project (onboarding speed, code readability, test posture, math correctness, and overall repo efficacy for agent and human contributors). Findings include concrete strengths, gaps, risks, and actionable ideas.

- The repo is largely self‑documenting about architecture and day‑to‑day practice, but not yet self‑documenting about top‑level success metrics for a 6‑month MSc thesis (no single “Project Charter” stating time‑boxed goals, milestones, and hard acceptance criteria). Adding an explicit charter would reduce ambiguity for new agents.
- README states the core objective crisply: “Minimal, fast‑to‑iterate stack for numerical experiments around the Viterbo conjecture,” emphasizing PyTorch‑first math, a CPU baseline, and a C++ extension scaffold; this aligns with a pragmatic thesis execution model (fast loops first, escalate to C++ only for hotspots).
- The “skills” system in AGENTS.md provides targeted, high‑signal onboarding for agents; this is a strong design principle for scaling many contributors: each task references only the skills needed, keeping cognitive load low.
- The repository offers a tight “quick loop”: `just sync` → `just checks` → `just ci`, which fosters short feedback cycles and supports the principle of fast, incremental progress.
- Devcontainer guidance is explicit with two variants (local vs Codespaces) and no default; this nudges conscious selection and helps avoid accidental mismatches. Clear pros/cons tradeoff is implicitly documented via owner workflow notes.
- CI builds docs (strict) and runs smoke‑tier tests on CPU‑only Torch wheels; this enforces documentation freshness and portability. Useful for a thesis context where doc drift can derail advisor review.
- The Justfile defines both local (uv‑env) and system‑Python CI paths; in CI, CPU‑only Torch is pinned with `torch==2.5.1` and policy env vars, ensuring deterministic wheels and predictable runtimes.
- The “Inner/CI/Deep” cadence is partially quantified in a brief (Inner <2 min, CI <5 min). This is great; however, onboarding speed targets (time to get first green checks, time to first successful PR/test run) are not explicitly stated or measured. Consider adding explicit “onboarding SLOs.”
- Testing posture is smoke‑first with an optional deep tier and benchmark smoke tier. Selection/incremental execution is wired (scripts/inc_select.py), which supports fast local runs; tests are deterministic with explicit seeds and float64 defaults in math.
- Coverage is available locally via `just coverage` but not enforced in CI (no thresholds, no failure on drops). For project‑wide quality, consider adding a modest coverage floor for critical math modules.
- Pyright “basic” runs in CI; there’s a strict config available but unused by default. A design principle could be to enforce strict typing for math/public APIs while keeping experiments looser.
- Lint is Ruff‑based with a strong, correctness‑oriented ruleset; docstring style is Google. Tests are exempt from docstring rules. This balances signal and practicality.
- Layering is explicit and enforced by convention: `math` (pure) ← `datasets` (adapters) ← `models` (experiments) with `_cpp` extensions plus Python fallbacks. This is a crucial principle for readability and long‑term maintainability.
- Math purity principle is consistently applied: functions are Torch‑first, preserve dtype/device, avoid I/O/state, and accept caller devices. Tests assert shapes/dtypes/tolerances directly.
- Device policy is explicit: CPU baseline everywhere; GPU is optional and confined to `models/`. This helps reproducibility and keeps CI lean.
- Numerics policy is explicit: math defaults to float64; explicit tolerances scale like `sqrt(eps)`. This is called out in docs and reflected in tests.
- Docs quality is high for math modules: MkDocs site has a dedicated Math API section with shape/dtype contracts and known limitations; architecture overview explains layering and invariants; oriented‑edge spectrum has a detailed spec and warnings about heuristics.
- The oriented‑edge R^4 solver is guarded with a wall‑clock budget via `enforce_time_budget` (default 30s); this is a strong principle for keeping exploration bounded and repeatable. Good for agents and CI stability.
- C++ extensions are wrapped in lazy loaders with Python fallbacks and a “safe exceptions” policy; the design degrades gracefully if build fails, ensuring portability and preventing local‑only breakage.
- Tests cover: imports/smoke, core polytope queries and H/V conversions, volume in 1D/2D/3D, datasets+collate, minimal capacity checks (2D and certain 4D structures), C++ extension fallbacks, and performance smoke. Breadth is solid for a starting point.
- Several tests are “skip‑documented” to capture planned backends (triangulation, Lawrence, Monte Carlo) and certify the target verification strategy; this documents intent and supports roadmap discussions.
- Datasets and collators are designed for ragged data patterns (list vs padded+mask), with deterministic test coverage; this supports robust batching and model experimentation without leaking complexity into math.
- Notebook policy favors Jupytext `.py` notebooks with metadata preservation; execution and artefact policies are clear (store in `artefacts/`, not in Git). This reduces repo noise and supports reproducibility.
- Project owner workflow (VibeKanban + Devcontainer + Remote) is extensively documented and operationally precise (ports, auth, caches, bind mounts). Critical for coordinating many agents in a thesis timeline.
- VibeKanban is the single backlog; tasks are expected to be short, self‑contained, with acceptance criteria and links into docs. This supports a “many small increments” principle.
- Skills metadata and AGENTS.md sections are auto‑managed by a script (`scripts/load_skills_metadata.py`) and checked by lint; this keeps onboarding references fresh and consistent.
- The math docs include rigorous specifications (e.g., `cstar_constant_spec.md`) and clear references; this pushes a thesis‑appropriate standard of correctness and traceability.
- Thesis/Overleaf structure is included (`thesis/` with main.tex, chapters, bib). However, thesis meta‑guidance (chapter plan, milestones, weekly targets) is not captured beyond placeholders; consider adding a “thesis plan” doc that aligns code milestones to chapter milestones.
- The README’s “Layout” and “Architecture” pointers are concise and accurate; they cross‑link to AGENTS.md and architecture docs, reinforcing a self‑service onboarding loop.
- `vertices_to_halfspaces` uses SciPy’s ConvexHull (CPU float64) then maps back to caller dtype/device; dependency is declared; tolerances and deduplication are handled explicitly. Good engineering choices for correctness over raw speed in early stages.
- The volume implementation mixes specialized low‑dimensional formulas with a facet‑height accumulation for D≥3; tests cover 1D/2D/3D and emphasize dtype behavior and tolerance control; higher‑dimensional exact backends are stubbed with skip‑tests (roadmap captured).
- Capacity algorithms module contains 2D placeholders, 4D specialised paths (Lagrangian products and oriented‑edge spectrum), and a hybrid primal–dual guard asserting consistency; tests exercise consistency across implementations and error handling (odd dims, degenerate facet counts).
- Deterministic RNG: datasets use an internal `torch.Generator` and fixed seeds in tests; math test modules set `float64` defaults explicitly. This enforces reproducibility across runs/machines.
- Command runner (Justfile) offers a coherent mental model: format/lint/type/test, plus coverage and docs; CI variants avoid uv’s virtualenv overhead by switching to system Python for speed.
- CI concurrency and cache usage (uv cache) are configured; docs deploy to GitHub Pages with strict checks and link validation. This helps avoid broken links in advisor‑facing docs.
- Pyright is invoked at the `src/viterbo` target; repository‑wide strictness is optional; this avoids noise while still catching egregious issues. A future tightening policy can start with the math layer only.
- Pre‑commit config includes a pre‑push pytest hook referencing JAX env vars from an earlier era; harmless but noisy/incongruent with PyTorch‑first design. Consider pruning/changing to a Torch‑aligned, smoke‑tier quick run.
- No CONTRIBUTING.md; AGENTS.md is used instead. For human external contributors, a minimal CONTRIBUTING stub could point to AGENTS.md and Justfile commands.
- Public API surfaces remain module‑scoped (no re‑exports); explicit imports and no `__all__` indirection aligns with readability and stability principles.
- Naming is semantics‑first (`normals`, `offsets`, `support`, etc.); docstrings follow Google style and emphasize shapes/dtypes/invariants, matching the “math purity” and correctness principles well.
- Tests include rich docstrings that describe the invariant (e.g., smoke vs deep), improving auditability and serving as living documentation.
- The papers/ folder includes curated TeX sources and notes; while heavy, it centralizes references and aligns with the thesis’s research nature. Good for on‑ramp context.
- MkDocs nav is concise and avoids overwhelming users; “Math API” is discoverable and central for implementers.
- The repo enforces “no implicit device moves” and CPU/float64 norms in math; this keeps CI portable and numerics predictable; models retain the freedom to use GPU when needed.
- Benchmarks isolate to smoke/deep tiers and require pytest‑benchmark; they’re skipped otherwise, keeping default runs lean and deterministic.
- The `_cpp` scaffold demonstrates both single‑file and multi‑file builds, with lazy load and `USE_NINJA=0` fallback; tests validate presence/absence and fallback correctness. This offers a safe template for future hotspots.
- Incremental test selection (`scripts/inc_select.py`) is provided but not yet deeply documented; still a nice win for agent productivity in larger change sets.
- `pytest.ini` sets `DeprecationWarning` to error; this is a correctness‑first posture, preventing silent drift as dependencies evolve.
- Type stubs under `typings/` cover third‑party gaps (jax remnants present); useful if needed by typechecker; residual JAX references can be further trimmed for clarity.
- Docs call out pure‑math semantics while omitting implementation detail; this separation is appropriate and supports both implementers and thesis writing by keeping the spec/code split clear.
- The environment requires Python ≥3.12; uv is a mandatory tool; this is fine, but first‑time contributor friction could be reduced by a one‑liner bootstrap script or more prominent mention in README’s Quickstart (it exists, but an install helper would help non‑agent contributors).
- There is no explicit “performance budget” or baseline targets documented (e.g., max acceptable runtime for critical solvers on reference shapes). Tests use timeouts and the runtime budget decorator, but success criteria for performance improvements/regressions are not formalized.
- Error messages in math are descriptive and early‑failing (shape, dtype, device mismatches); good for onboarding and debugging.
- Dtype/device propagation is consistent; downcasts are avoided by policy; where conversions occur (e.g., SciPy hull), returns are re‑cast appropriately. This reduces numerics bugs.
- There is a clear policy against implicit IO/stateful caches in math and C++; consistent with deterministic, testable behavior.
- The “thesis topic” brief outlines objectives (systolic ratio in 4D, benchmarking counterexamples, exploring parametric families) and acceptance points; helpful for alignment with the academic deliverable.
- Risk: higher‑dimensional exact volume/capacity paths are still stubbed; if they are core to thesis goals, schedule guardrails and staged milestones are needed (e.g., 2D parity → 4D Lagrangian products → general 4D with CH budgets → general 2n with relaxations).
- Risk: oriented‑edge solver exposes tunables (memo/budgets/rotation cap) with safety caveats; tests cover consistency and error modes, but an explicit “certified vs heuristic” policy table would help operators choose safe defaults per experiment.
- Risk: Deep tests are minimal; they exist primarily to validate tier wiring. As math expands, beef up deep tier with invariants and stress cases (e.g., near‑degenerate faces, ill‑conditioned normals) to reduce future regressions.
- Risk: No CI job enforces documentation build on pull requests with a failing threshold for warnings across all pages; current docs workflow is good, but adding strict link checks (present) plus a budget for warnings (currently strict) is already effective.
- Risk: No explicit non‑functional goals captured (e.g., “CI must complete in <7 minutes on GitHub hosted runners,” “onboarding to first passing `just checks` must be <30 minutes on a clean machine”). Consider adding these as measurable SLOs.
- Suggestion: Add a top‑level “Project Charter” doc with (a) thesis objectives, (b) 6‑month milestones, (c) success metrics (math correctness criteria, test coverage targets, CI time budgets, doc freshness), and (d) governance (how decisions are escalated or revised).
- Suggestion: Add a one‑liner bootstrap (`curl | bash` or a short script) to install uv and run `just setup`, primarily for humans; link from README.
- Suggestion: Move the pre‑push pre‑commit hook to a Torch‑aligned quick test or remove it to reduce noise; document preferred “pre‑commit vs just precommit-slow” usage.
- Suggestion: Gate `src/viterbo/math/**` under Pyright strict in CI to tighten correctness for public math APIs while keeping models and scripts at basic or optional strict.
- Suggestion: Introduce a minimal coverage threshold (even 50–60%) for `src/viterbo/math/` to prevent accidental erosion as the codebase grows; keep smoke‑tier jobs fast.
- Suggestion: Add a small “onboarding checklist” (time to green checks, time to run the demo notebook, time to run smoke tests) and capture actual median timings from fresh clones; report in README or AGENTS.md to set expectations.
- Suggestion: Publish a short “math invariants” page (or expand existing docs) summarizing global invariants: even‑dim dims for symplectic helpers, default dtype/device policy, tolerance scale, and deterministic RNG practices.
- Suggestion: For oriented‑edge solver, include a micro‑guide on safe defaults (e.g., budgets on, memo off for certification; budgets+memo on for exploration) and typical runtime expectations for the provided standard polytopes.
- Suggestion: Add small “reference instances” and expected runtimes to docs (e.g., hypercube, cross‑polytope, pentagon product); pair with acceptance thresholds to prevent unbounded regressions.
- Suggestion: Thesis directory could include a “milestones.md” outlining planned content delivery by week/month and mapping code deliverables to chapter sections; improves advisor communication and internal alignment.
- Suggestion: Consider a tiny CONTRIBUTING.md that points newcomers to AGENTS.md, Justfile commands, and docs nav; reduces confusion for GitHub visitors.
- Suggestion: Keep a section in AGENTS.md summarizing “measurable success targets” (onboarding, CI duration, docs build strictness, coverage floor) so agents have a single, actionable checklist.
- Suggestion: Make `skills/always.md` explicitly ask agents to record SLO deviations (“Needs‑Unblock: timing‑SLO”) when checks take too long; close the loop on performance hygiene.
- Suggestion: Confirm all legacy JAX remnants are pruned (typings and hooks) to remove ambiguity; current presence is benign but confusing in a PyTorch‑first project.
- Suggestion: Add a simple smoke test that exercises docs code fences or a subset of examples to prevent drift between docs and code samples (some projects run doctests selectively).
- Overall: The repository is in strong shape for rapid onboarding and iterative, correctness‑minded development. The main gap against “Project Design Principles” for a 6‑month MSc is the absence of explicit, measurable top‑level success criteria and a thesis milestone plan. Adding those would make the repo fully self‑documenting with respect to “what we aim to achieve” and “how we will know we’ve succeeded.”


Actions (first pass)
- Deferred; to be extracted during consolidation at the end of the review batch.

Cross-References
- README.md
- AGENTS.md
- Justfile
- skills/good-code-loop.md
- skills/testing-and-ci.md
- docs/charter.md

Validation
- Built docs strictly: `uv run mkdocs build --strict` — OK

Limitations
- Did not run full `just ci`; this pass focuses on principles and onboarding.
- Not a deep content review of math/algorithms; see Topics 6 and 7.

Status
- Finalized
