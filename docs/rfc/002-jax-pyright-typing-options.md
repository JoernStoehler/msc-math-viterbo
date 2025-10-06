# RFC 002 — Pyright strategies for JAX-powered geometry modules

- **Status**: Draft
- **Last updated**: 2025-10-06
- **Owners**: Geometry quantity maintainers
- **Related context**: JAX‑first geometry modules and wrappers:
  `src/viterbo/geometry/halfspaces/reference.py`, `src/viterbo/geometry/halfspaces/fast.py`,
  `src/viterbo/geometry/volume/_shared.py`, and SciPy interop under `src/viterbo/_wrapped/`.

## 1. Problem statement

The geometry restructure introduced JAX‑first implementations with reference/fast variants and
centralized SciPy interop via wrappers. These modules currently run under Pyright's global `strict`
mode with missing-imports escalated to errors. While the present code paths rely on `jax.numpy`
linear algebra and return JAX arrays by default (converting only at wrapper boundaries), further
work will inevitably require transformations such as `jax.jit`, `jax.vmap`, and structured PyTree
utilities. JAX does not ship comprehensive type stubs, so each incremental feature unlock risks
surfacing new `reportMissingTypeStubs` and `reportUnknownMemberType` errors in strict
mode.【F:pyrightconfig.json†L1-L7】

## 2. Options considered

### 2.1 Maintain local JAX stub packages

Create a repository-local `stubs/jax/` tree covering the precise subset of `jax`, `jax.numpy`, and
`jax.numpy.linalg` that the project touches, and point Pyright at it via `stubPath`. For
higher-level constructs (e.g., `jax.jit`), provide callable protocols that match our usage patterns
and keep signatures narrow.

- **Pros**
  - Preserves repo-wide `strict` typing guarantees without waivers.
  - Allows us to model return types that map cleanly onto downstream usage while retaining JAX
    arrays across the codebase.
  - Scales incrementally: only add stubs as the geometry modules adopt new primitives.
- **Cons**
  - Up-front busywork to mirror even the small API surface we already touch (`jax.numpy`, `linalg`,
    `jax.Array` aliases).
  - Requires ongoing maintenance to track signature changes when upgrading JAX releases noted in
    `pyproject.toml` (currently `jax>=0.4.31`).【F:pyproject.toml†L9-L15】
  - Complex decorators like `jax.jit` or `jax.vmap` need careful callable protocols to avoid false
    positives; mistakes silently erode type safety.
- **Maintenance cost**: Medium. Expect small updates per release cycle plus occasional refactors
  when we adopt new primitives.
- **Risk profile**: Low. Type coverage remains explicit and testable in CI.

### 2.2 Relax Pyright strictness or configure module-level escapes

Options include dropping the global `typeCheckingMode` to `standard`, toggling
`reportMissingTypeStubs` to `warning`, or marking the JAX packages as
`reportMissingImports: warning` via `pyrightconfig.json` overrides.

- **Pros**
  - Minimal engineering effort; configuration-only change.
  - Eliminates the immediate friction when adding new JAX features.
- **Cons**
  - Contradicts the repository's strict typing policy codified in `/AGENTS.md`, reducing reviewer
    confidence across _all_ modules.
  - Makes it harder to spot genuine regressions in our NumPy/SciPy code because strict diagnostics
    disappear globally.【F:AGENTS.md†L24-L56】
  - Module-level overrides propagate to future contributors, encouraging ad-hoc relaxations instead
    of disciplined typing.
- **Maintenance cost**: Low.
- **Risk profile**: High. Weakening static guarantees undermines a core guardrail for the thesis
  timeline.

### 2.3 Rely on targeted `typing.cast` and `type: ignore` hints

Keep Pyright strict globally but apply localized suppressions or explicit casts around JAX calls
when diagnostics appear.

- **Pros**
  - Tight scope: only the lines that trigger false positives are affected.
  - No tooling changes required; works immediately with upstream JAX packages.
- **Cons**
  - Accumulates noise in performance-critical kernels, obscuring real typing mistakes.
  - Inline suppressions require TODOs per repository policy, so reviewers must track debt
    manually.【F:AGENTS.md†L36-L54】
  - Provides no reusable abstraction for future contributors; each new JAX primitive repeats the
    workaround.
- **Maintenance cost**: High. Debt scales linearly with JAX usage and must eventually be repaid.
- **Risk profile**: Medium. Local ignores can mask legitimate bugs if copied broadly.

## 3. Decision

**Recommended path**: Proceed with option 2.1 (local stub packages) and pair it with thin, typed
wrapper functions. Keep JAX arrays as the default across modules; confine NumPy/SciPy usage to
`viterbo._wrapped/`. This keeps the strict typing contract intact while constraining the stub
surface to what our geometry variants actually expose. Wrappers can normalise return types to NumPy
arrays and centralise any necessary `typing.cast` calls, further reducing duplication.

## 4. Follow-up actions

1. Scaffold `stubs/jax/` with minimal coverage for `jax`, `jax.numpy`, and `jax.numpy.linalg` used
   today.
1. Add a `stubPath` entry to `pyrightconfig.json` once the stub tree exists.
1. Introduce wrapper helpers in `_shared` modules for future JAX functionality (`jit`, `vmap`) so
   that typed call sites remain small and auditable.
1. Document the stub maintenance workflow (when to extend, how to sync with JAX release notes) in
   the geometry contributor guide.

## 5. Deferred alternatives

- Revisit configuration relaxations if maintaining stubs becomes unmanageable due to upstream churn.
- Investigate community-maintained stub packages once they offer the strictness guarantees we
  require.
