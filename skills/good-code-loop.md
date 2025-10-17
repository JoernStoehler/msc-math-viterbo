---
name: good-code-loop
description: Use for shipping correct code with tests while preserving architecture boundaries and PR hygiene.
last-updated: 2025-10-17
---

# Good Code Loop

## Instructions
- Start from the VibeKanban task, then run `uv run python scripts/load_skills_metadata.py` and load this skill.
- Work in small increments; keep math pure, uphold layering, and keep tests close to changes.
- Use the command palette below; capture deviations and escalate architectural or performance risks early.

## Inner Loop

1. Run `just checks` for the fast gate (format, lint, type, smoke tests).
2. Use `just test` for focused smoke runs; add `INC_ARGS="--debug"` to inspect selection.
3. Before PR or after substantial changes, run `just ci` for parity.
4. If Ruff auto-fixes are available, `just fix`; rerun `just checks`.

## Architecture Boundaries

- Respect strict layering: `math` ← `datasets` ← `models`. Never import upward. C++ lives in `_cpp/` with Python fallbacks when feasible.
- Keep math-layer APIs pure and device-agnostic: no I/O, no globals, accept caller devices, return tensors.
- Side effects and batching live in datasets/models; math remains tensor-native.
- Escalate with `Needs-Unblock: architecture` when crossing layers, adding dependencies, or changing asymptotics.

## Coding Standards Highlights

- PyTorch-first public APIs; document dtype/shape explicitly.
- Absolute imports; no wildcards or re-export indirection.
- Google-style docstrings emphasizing semantics, invariants, units, shapes.
- Use vectorized ops (`torch.linalg`, broadcasting) over Python loops.

## Precision, Randomness, Devices

- Math defaults to `float64`; models default to `float32`. Document deviations.
- Handle RNG via `torch.Generator`; avoid int seeds except at boundaries (document conversions).
- Do not move tensors across devices implicitly; callers control `.to(device)`.

## Testing Patterns

- Favor invariants/property tests for geometry; seed generators for determinism.
- Use `torch.testing.assert_close`, `pytest.approx`, or `math.isclose`; avoid raw float equality.
- Tests must cover ragged data when applicable; use representative inputs instead of exhaustive permutations.

## PR & Commits

- Conventional Commits (`type(scope): summary`).
- Include validation evidence (Ruff/Pyright/Pytest; benchmarks if relevant) in the PR body.
- Leave `NotImplementedError` placeholders unmasked; avoid try/except to hide missing code.
- Run `just ci` before handoff or PR creation for parity.

## Review Checklist

- [ ] Layering respected; no upward imports or hidden state.
- [ ] Public APIs document shapes/dtypes; device semantics explicit.
- [ ] No implicit device hops or silent dtype casts.
- [ ] Randomness via `torch.Generator` where applicable.
- [ ] Tests cover representative paths and invariants.

## Related Skills

- `data-collation` — batching/ragged data policies and helpers.
- `math-layer` — detailed math implementation guidance.
- `performance-discipline` — measuring and addressing bottlenecks.
- `basic-environment` — golden commands and repo navigation.

