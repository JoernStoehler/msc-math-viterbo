---
name: coding-standards
description: Apply project-wide coding conventions with emphasis on math purity, Torch usage, and architectural layering.
last-updated: 2025-10-17
---

# Coding Standards

## Core Principles

1. Keep `src/viterbo/math/` pure: no I/O, no hidden state, accept caller devices, return tensors.
2. Observe layering: `math` ← `datasets` ← `models`; lower layers never import higher ones.
3. Default to Torch tensors in public APIs; document dtype and shape assumptions explicitly.
4. Use absolute imports; avoid wildcard imports and re-export indirection.
5. Follow Conventional Commits when preparing commit messages.

## Implementation Guidelines

- Precision:
  - Math utilities use `float64` unless documented otherwise.
  - Models default to `float32`; state deviations clearly in docstrings.
- Ragged data handling:
  - Accept Python lists of tensors or padded tensors with mask arguments.
  - Provide `collate_fn` helpers in `datasets` for DataLoader integration.
- Randomness:
  - Pass `torch.Generator` objects instead of integer seeds.
  - Do not move tensors across devices implicitly; let callers decide.
- Docstrings:
  - Use Google style, focus on semantics, invariants, units, and shapes.
  - Omit redundant shape asserts unless solving a specific bug.
- Typing:
  - Prefer built-in generics (`list[str]`, `dict[str, torch.Tensor]`).
  - Avoid custom typedefs for shapes/dimensions.

## Review Checklist

- [ ] Module respects stacking order and does not reach into higher layers.
- [ ] Functions clarify tensor shapes/dtypes where non-obvious.
- [ ] Tests rely on `torch.testing.assert_close` or `pytest.approx` for numeric checks.
- [ ] No silent downcasts or implicit device hops.
- [ ] Added comments are minimal and clarify intent rather than restating code.

## Related Skills

- `testing-workflow` — for validation steps that accompany code changes.
- `repo-onboarding` — ensures metadata and planning policies are satisfied before coding.
