# Single JAX‑First LP Solver

- Status: Scheduled
- Last updated: 2025-10-05
- Owner / DRI: Unassigned
- Reviewers: Maintainer (PI)
- Related docs: AGENTS.md, docs/tasks/02-task-portfolio.md

## 1. Context and intent

The repository currently uses a SciPy-backed wrapper and an abstraction layer (`LinearProgram`,
backends) for linear programs. This adds cognitive overhead for new contributors and contradicts the
stated JAX‑first policy. We want a single, simple LP entrypoint that is JAX‑native, composable with
`jit`/`vmap`, and easy to onboard.

We will standardize on JAXopt’s OSQP as the JAX‑first solver. LPs are a special case of QP with
`Q = 0`, and OSQP fits the project’s JAX‑first constraints and type policies.

## 2. Objectives and non-goals

### In scope

- Provide a single public function `linprog_jax` that:
  - Accepts `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq`, `bounds` with SciPy-like semantics.
  - Validates shapes/dtypes (float64 default), returns JAX arrays.
  - Maps constraints to OSQP form `l <= A x <= u` with `Q = 0`.
  - Returns `(x, fun)` and raises precise errors on non-optimal termination.
- Remove the LP abstractions and SciPy wrapper.
- Update tests to cover feasible/infeasible and bounds-normalization cases.
- Update public API exports and a short README note.

### Out of scope

- Adding alternative MILP/QP backends.
- Performance optimizations beyond correctness.
- Reworking geometry or symplectic modules.

## 3. Deliverables and exit criteria

- Code:
  - `viterbo.optimization.linprog_jax` implemented (JAXopt OSQP, `Q = 0`).
  - Deleted files and APIs:
    - `src/viterbo/optimization/solvers.py` (all LP abstractions)
    - `src/viterbo/_wrapped/optimize.py` (SciPy linprog wrapper)
  - Public API updated to export only `linprog_jax`.
- Tests:
  - New tests `tests/viterbo/optimization/test_linprog_jax.py` covering:
    - Feasible LP with equality + inequality + bounds.
    - Bounds normalization (None/±inf/scalars).
    - Invalid shapes → `ValueError`.
    - Infeasible system → `RuntimeError`.
- Docs:
  - Short README snippet: “LP solver policy: use `viterbo.optimization.linprog_jax`”.
- Tooling:
  - `make ci` green (Ruff format/lint, Pyright strict, tests).

## 4. Dependencies and prerequisites

- `jaxopt` is already included in `pyproject.toml` (golden path). Confirm the version meets the
  minimal requirement; no environment change needed.
- No change to SciPy usage elsewhere (`_wrapped/spatial.py` remains).

Blocking prerequisites: none; defer execution until currently queued PRs land.

## 5. Execution plan and checkpoints

1. Implement `linprog_jax` (JAX arrays in/out, shape checks, constraint builder to `A, l, u`, OSQP
   call with sensible defaults, clear status mapping).
1. Delete LP abstractions and SciPy wrapper; remove all re‑exports.
1. Update `optimization/__init__.py` and package `__init__.py` to export only `linprog_jax`.
1. Replace solver tests with `test_linprog_jax.py` and adjust assertions.
1. Add the README policy note.
1. Run `make ci`; iterate until green.

Checkpoint: single PR with focused diff (preferably ≤300 LOC) and CI summary in the PR description.

## 6. Effort and resource estimates

- Agent time: Low → Medium (one PR, code + tests + docs).
- Compute budget: Low (unit tests only).
- Maintainer involvement: Low (review, environment approval for `jaxopt`).

## 7. Testing, benchmarks, and verification

- CI: `make ci` (format → lint → typecheck → tests) must pass.
- Numerical checks use explicit tolerances: `rtol=1e-9`, `atol=1e-8`.
- Optional local timing vs. previous SciPy path (informational only, not required in CI).

## 8. Risks, mitigations, and escalation triggers

- Risk: OSQP performance on CPU may be slower than HiGHS.
  - Mitigation: accept for JAX‑first simplicity; revisit if regression impacts core workflows.
- Risk: Status mapping/edge cases for infeasible/unbounded problems.
  - Mitigation: translate non‑optimal OSQP statuses to `RuntimeError` with clear messages; add unit
    tests.
- Escalate (Needs-Unblock) if:
  - Adding `jaxopt` clashes with the devcontainer or CI setup.
  - Material performance regression (>10%) blocks progress on downstream tasks.

## 9. Follow-on work

- Optional: batched solve helper using `vmap` for multiple LPs with shared structure.
- Optional: benchmark note comparing OSQP vs. HiGHS for representative LPs.
- Consider re-expressing EHZ DP core in JAX control flow for jit when/if it is a bottleneck.
