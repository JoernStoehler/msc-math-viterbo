# Tasks Workflow (Parallel Development)

Briefs in this folder represent the active backlog. Each file tracks scope,
dependencies, and status using YAML front matter. Treat `status` as the
assignment signal:

- `draft` / `proposed` — ready for adoption. When you begin implementation,
  update the brief to `status: in-progress` inside your feature branch so the
  PR advertises ownership.
- `in-progress` — an agent already owns the task on a feature branch. Update
  the brief to `status: completed` (or revert the change) before the branch is
  merged so `main` stays free of in-progress markers.
- `completed` — work merged to `main`; move the brief to `docs/tasks/archived/`
  once the notes remain useful. Delete the file if it no longer adds context.

Follow the workflow from `AGENTS.md`, keeping math code pure and landing green
CI in each PR. Archive completed briefs promptly so this directory remains a
clean queue for new agents.

## Recently archived

- 2025-10-13 Math/Geometry: vertex transforms, volume, regular n-gon
- 2025-10-13 Math/H-Rep: conversions and transforms
- 2025-10-13 Math/Random: polytope generators (H and V)
- 2025-10-13 Math/Symplectic: J, random symplectic matrices, Lagrangian product
- 2025-10-13 Define smoke tests for core algorithms
- 2025-10-13 Datasets: AtlasTiny baseline dataset
- 2025-10-13 Datasets: Collate edge cases and ragged batches
- 2025-10-13 CI: enforce CPU-only Torch index configuration
- 2025-10-13 C++ harness and documentation
- 2025-10-13 Math/Symplectic: EHZ capacity (H/V) and minimal action cycle

## 2025-10-14 Triage

- Archived historical scaffolding briefs whose implementations now live in
  `src/viterbo/math/` (see `docs/tasks/archived/`).
- Confirmed the active backlog focuses on higher-dimensional capacity solvers
  plus dataset follow-ups.
- Relocated superseded workflow briefs under `docs/briefs/archive/` to avoid
  conflicting instructions.

### Active backlog snapshot

- `2025-10-13-task-math-counterexample-loader.md` *(proposed)* — dataset for
  repro cases around Minkowski billiards.
- `2025-10-14-task-datasets-atlas-tiny-4d.md` *(draft)* — extend AtlasTiny to 4D
  Lagrangian products.
- `2025-10-14-task-math-4d-capacity.md` *(draft)* — generalise EHZ capacities
  beyond the planar solver.

### Follow-ups worth tracking

- [ ] Keep `docs/tasks/` limited to adoptable briefs; archive or delete files
      immediately after their implementation merges.
- [ ] Periodically prune the archived directory when a brief no longer adds
      context worth keeping.
- [ ] Keep future ideas in `docs/tasks/future/` and promote them only when
      scoping work begins.
