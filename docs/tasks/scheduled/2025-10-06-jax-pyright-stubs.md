- **Status**: Scheduled
- **Last updated**: 2025-10-06
- **Owner / DRI**: Unassigned
- **Reviewers**: Maintainer
- **Related docs**: [RFC 002](../../rfc/002-jax-pyright-typing-options.md), [Geometry module refactor brief](../completed/2025-10-04-geometry-module-refactor.md)

## 1. Context and intent
Pyright strict mode currently rejects imports and decorators from JAX, blocking further work on the geometry quantity variants. RFC 002 recommends providing repository-local type stubs and lightweight wrappers so strict type checking remains effective without suppressions. This task schedules the stub tree implementation and any supporting helpers needed to make the geometry modules type-clean.

## 2. Objectives and non-goals

### In scope
- Provide a `stubs/jax/` package that covers the subset of APIs used in `src/viterbo/geometry`.
- Ensure Pyright strict passes across the repository without inline `type: ignore` directives for JAX usage.
- Document the stub maintenance workflow and integration points in developer-facing docs.

### Out of scope
- Expanding geometry functionality beyond existing behaviour.
- Introducing new runtime dependencies or altering the golden-path environment.
- Resolving broader JAX-on-device or GPU support questions.

## 3. Deliverables and exit criteria
- A committed stub hierarchy (e.g., `stubs/jax/__init__.pyi`) with coverage for `jax`, `jax.numpy`, `jax.lax`, and decorators invoked in the codebase.
- Updated `pyrightconfig.json` (or equivalent) to include the new stub path.
- Passes `uv run pyright` without JAX-related diagnostics.
- Contributor documentation describing how to extend the stubs when new APIs are adopted.

## 4. Dependencies and prerequisites
- Final decision recorded in [RFC 002](../../rfc/002-jax-pyright-typing-options.md).
- Access to existing geometry modules to inventory required JAX symbols.
- Coordination with maintainers if the stub path needs CI configuration adjustments.

## 5. Execution plan and checkpoints
1. Inventory JAX symbols used across `src/` and `tests/`, prioritising geometry modules.
2. Generate `.pyi` stubs for the needed modules, modelling signatures on official docs and existing type hints.
3. Integrate the stubs via `pyrightconfig.json` and run Pyright to validate coverage.
4. Draft documentation updates (e.g., contributor guide or geometry README) summarising stub usage and maintenance expectations.
5. Final review to ensure no runtime changes slipped in and that strict type checking remains green.

## 6. Effort and resource estimates
- Agent time: Medium (est. 1-2 focused sessions).
- Compute budget: Low (type checking and unit tests only).
- Expert/PI involvement: Low (review of stub coverage if needed).

## 7. Testing, benchmarks, and verification
- Mandatory: `uv run pyright` after stubs integrate.
- Optional: `uv run pytest tests/geometry -q` to guard against regressions during refactorings that introduce typed wrappers.
- No performance benchmarks required.

## 8. Risks, mitigations, and escalation triggers
- **Risk**: Stub drift when JAX APIs change → **Mitigation**: Document update workflow and include routine validation in task follow-ups.
- **Risk**: Missing decorator behaviours (`jax.jit`, `jax.vmap`) → **Mitigation**: Provide typed wrappers or protocol definitions as needed; escalate if semantics unclear.
- **Escalation triggers**: Pyright still reports diagnostics after stub integration, or the task reveals broader dependency configuration issues.

## 9. Follow-on work
- Implement helper wrappers for higher-order decorators (`jax.jit`, `jax.vmap`) if upcoming tasks require richer typing.
- Evaluate automated checks (e.g., stub-mypy sync) once the stub tree stabilises.
