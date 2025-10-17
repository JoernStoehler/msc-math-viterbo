---
name: coding-standards
description: Apply project-wide coding conventions with emphasis on math purity, Torch usage, and architectural layering.
last-updated: 2025-10-17
---

# Coding Standards

## Core Principles

1. Keep `src/viterbo/math/` pure: no I/O, no hidden state, accept caller devices, return tensors.
2. Observe layering: `math` ← `datasets` ← `models`; lower layers never import higher ones. PyBind extensions live in `_cpp/` with Python fallbacks when appropriate.
3. Default to Torch tensors in public APIs; document dtype and shape assumptions explicitly.
4. Use absolute imports; avoid wildcard imports and re-export indirection. No `__all__` indirection.
5. Follow Conventional Commits for commit messages, even on agent branches.

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
