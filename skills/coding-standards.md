---
name: coding-standards
description: This skill should be used when applying project-wide coding conventions for PyTorch-first design, math purity, and strict layering.
last-updated: 2025-10-17
---

# Coding Standards

## Instructions
- Before coding, load this skill and confirm target layer responsibilities; keep `math` pure and adhere to strict layering.
- Document tensor shapes/dtypes in public APIs; prefer Torch tensors in inputs/outputs.
- Handle RNG via `torch.Generator`, avoid implicit device moves or silent dtype casts, and write representative tests.
- Run `just checks` locally; escalate if a change alters layer boundaries or introduces C++ without a clear hotspot.

## Core Principles

1. Keep `src/viterbo/math/` pure: no I/O, no hidden state, accept caller devices, return tensors.
2. Observe layering: `math` ← `datasets` ← `models`; lower layers never import higher ones. PyBind extensions live in `_cpp/` with Python fallbacks when appropriate.
3. Default to Torch tensors in public APIs; document dtype and shape assumptions explicitly (“PyTorch-first”).
4. Use absolute imports; avoid wildcard imports and re-export indirection. No `__all__` indirection.
5. Honour Google-style docstrings and keep names semantic (`normals`, `offsets`, etc.) to reflect geometry intent.

## Precision & Numerics

- Math utilities prefer `float64`; document any deviations.
- Models default to `float32` for training loops; surface dtype changes in docstrings.
- Avoid silent downcasts. If conversion is necessary, perform it explicitly and mention rationale.
- Use `torch.testing.assert_close`, `pytest.approx`, or `math.isclose` for numerical checks; avoid raw equality comparisons on floats.

## Ragged Data & Collation

- Accept Python lists of tensors or padded tensors with mask arguments for ragged data scenarios.
- Provide `collate_fn` helpers in `datasets` so DataLoaders can batch ragged inputs consistently.
- Keep math-layer utilities agnostic of batching; let datasets/models perform collation.

## Randomness & Devices

- Pass `torch.Generator` objects instead of integer seeds to preserve reproducibility.
- Do not move tensors across devices implicitly; let callers handle `to(device)` decisions.
- When bridging to external APIs that only accept seeds, document the conversion and constraints clearly.

## PyTorch & C++ specifics

- Assume CPU execution unless a task explicitly requests CUDA; use device-agnostic code paths by default.
- Keep math APIs pure even when backed by C++ extensions; they must accept tensors on the caller’s device without copying.
- When adding C++ kernels, provide Python fallbacks when feasible and document build requirements.
- Never introduce hidden global state (e.g., cached tensors) inside extensions; prefer explicit handles.
- Push plotting/IO conversions to call sites (e.g., `tensor.detach().cpu().numpy()`), keeping library APIs tensor-native.

## Docstrings & Commenting

- Use Google-style docstrings focusing on semantics, invariants, units, and shapes.
- Include shape/dtype expectations when non-obvious, including batch dimensions.
- Avoid redundant comments; reserve them for tricky invariants or algorithmic context.

## Typing & Imports

- Prefer built-in generics (`list[str]`, `dict[str, torch.Tensor]`, etc.) and union syntax (`float | torch.Tensor`) over custom typedefs.
- Future annotations are enabled project-wide; omit `from __future__ import annotations` where redundant.
- Order imports as standard library, third-party, then local. Rely on Ruff for formatting but keep groupings logical.
- No wildcard imports; each symbol should be explicitly named.

## Testing Alignment

- Mirror the testing policies in `skills/testing-workflow.md`: run `just checks` locally before requesting review.
- Tests should exercise representative usage paths rather than exhaustive permutations. Focus on invariants.
- Use property-style tests for geometry routines where invariants are critical; seed randomness explicitly.
- Prefer `torch.testing.assert_close`, `pytest.approx`, or `math.isclose` rather than equality checks; tune tolerances in docstrings.

## Process Expectations

- Follow Conventional Commits for all commit messages (`type(scope): summary`).
- Leave placeholders as bare `NotImplementedError`; do not mask them with `try/except` so missing code surfaces loudly.
- Avoid duplicating asserts solely to survive `python -O`; rely on core logic and dedicated validation tests.

## Review Checklist

- [ ] Module respects layering; no imports from higher-level packages.
- [ ] Public APIs document tensor shapes/dtypes where non-obvious.
- [ ] Randomness handled via `torch.Generator` or documented seeds.
- [ ] No implicit device hops or silent dtype conversions.
- [ ] Tests or examples cover ragged data handling when applicable.

## Related Skills

- `testing-workflow` — ensures validation accompanies code changes.
- `repo-onboarding` — aligns planning and metadata expectations.
- `performance-discipline` — governs optimization and profiling changes.
