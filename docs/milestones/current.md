status: active
last-updated: 2025-10-21
owners: [Project Owner Jörn]
---

# Current Milestone — Atlas Tiny + Math Core

## Scope right now
- **Math layer APIs.** Expect `viterbo.math.polytope`, `viterbo.math.constructions`, `viterbo.math.convex_hull`, `viterbo.math.polar`, `viterbo.math.similarity`, `viterbo.math.symplectic`, `viterbo.math.incidence`, and `viterbo.math.volume` to be present and documented (`docs/math/*.md`). The `viterbo.math.capacity_ehz.*` solvers (`algorithms`, `cycle`, `lagrangian_product`, `ratios`) and their Torch-first contracts are in scope.
- **Datasets + collation.** Atlas Tiny loaders (`viterbo.datasets.atlas_tiny*`) and ragged batching helpers under `viterbo.datasets.core` are ready for use. See `docs/datasets/atlas_tiny.md` for schema and invariants.
- **Runtime glue.** `viterbo.runtime` and the high-level smoke in `tests/test_smoke.py` define today’s supported entrypoints; nothing else is considered stable.
- **Docs.** Rely on `AGENTS.md`, this milestone page, the math module briefs, and `docs/workflows-project-owner-vibekanban.md` for owner/board mechanics. Anything outside those paths is optional background.

## Authoritative tests (smoke tier unless noted)
- **Core invariants.** `tests/test_smoke.py`, `tests/test_collate.py`, and `tests/test_cpp_extension.py` guard the baseline API and Python fallbacks.
- **Math layer.** Keep the polytope + symplectic suite green: `tests/math/test_polytope_smoke.py`, `tests/math/test_polytope.py`, `tests/math/test_polytope_library.py`, `tests/math/test_polytope_hull_properties.py`, `tests/math/test_polytope_representation.py`, `tests/math/test_symplectic.py`, `tests/math/test_symplectic_properties.py`, `tests/math/test_constructions.py`, `tests/math/test_volume_smoke.py`, `tests/math/test_minkowski_billiard.py`, and the minimal-action / capacity set (`tests/math/test_capacity_frontends.py`, `tests/math/test_capacity_ehz_haim_kislev.py`, `tests/math/test_capacity_oriented_edge.py`, `tests/math/test_oriented_edge_*`, `tests/math/test_minimal_action*`).
- **Datasets.** Atlas Tiny coverage (`tests/datasets/test_atlas_tiny.py`, `tests/datasets/test_atlas_tiny_io.py`) plus the benchmark plumbing in `tests/performance/test_atlas_tiny_build.py` (smoke marker via pytest-benchmark).
- **Selectors + markers.** `tests/test_deep_placeholder.py` and the benchmark files tagged with `benchmark`/`deep` stay skipped unless you opt into deeper tiers; do not delete them.

## Deferred / “ignore for now”
- **C++ surfaces.** Anything beyond the toy ops in `src/viterbo/_cpp` is out of scope; follow the decision tracked in [C++ Extensions Stance](https://vibekanban.joernstoehler.com/projects/dc72e60d-2885-4376-bdf6-00a5a2c5e24c/tasks/f14f8030-bcd4-428d-8be3-bf8572a99df2).
- **Extended selectors.** Selection ergonomics (`just test` defaults, deep-tier gating) are being reworked under [Simplify Dev Loop](https://vibekanban.joernstoehler.com/projects/dc72e60d-2885-4376-bdf6-00a5a2c5e24c/tasks/667ec958-3a3e-44ab-96fd-887709fd0fd4); do not rely on undocumented flags.
- **Status banners & doc polish.** Keep working from the core briefs; milestone messaging will land via [Docs Status Banners](https://vibekanban.joernstoehler.com/projects/dc72e60d-2885-4376-bdf6-00a5a2c5e24c/tasks/f7c7bd34-ba34-447d-aa7d-763d84ab0fce).
- **Unimplemented math modules.** Anything not listed above (e.g., capacity stubs in `viterbo.math.capacity_ehz.stubs`, prospective `_cpp` accelerators, additional dataset builders) is explicitly out until new milestones land.

## Tooling defaults
- Run `just checks` before handoff (Ruff format/lint, Pyright basic, smoke-tier pytest). For fast iterations use `just test` (impacted smoke); force a full smoke run via `just test-full`.
- Opt into deeper tiers only when scoped: `just test-deep` adds `pytest -m "smoke or deep"`. Benchmarks live behind `just bench`/`just bench-deep`; coordinate before recording numbers.
- CPU guardrails: Python processes inherit a 180s CPU cap from `sitecustomize.py`. Raise or disable with `VITERBO_CPU_LIMIT=<seconds>` (use `0` to turn it off) when running long notebooks or bulk datasets; reset the env var afterwards.
- Incremental selector hints: set `INC_ARGS="--debug"` with `just test` when you need to inspect which tests were picked; avoid editing `.cache/inc_graph.json` manually.

## Task intake checklist
- Read `skills/repo-onboarding.md` for environment setup and command quickstart.
- Refresh `skills/good-code-loop.md` to stay aligned on coding + review expectations.
- Skim `skills/testing-and-ci.md` for tier definitions and selector mechanics.
- Check `skills/vibekanban.md` before updating tickets or escalating blockers.
- Use `skills/writing.md` when producing milestone notes, PR descriptions, or advisor updates.
