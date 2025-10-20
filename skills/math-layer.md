---
name: math-layer
description: This skill should be used when implementing or modifying math-layer geometry utilities in `src/viterbo/math`.
last-updated: 2025-10-18
---

# Math Layer Workflow

## Instructions
- Keep math modules pure: accept caller devices, avoid I/O/global state, and return tensors.
- Document dtypes/shapes; prefer `float64` unless noted, and accept `torch.Generator` for reproducible randomness.
- Write invariance/property tests and run `just checks` locally; avoid implicit device moves or silent dtype casts.
- Escalate for asymptotic changes, naming deviations, or new C++ dependencies.

## Scope

Covers implementing or modifying utilities under `src/viterbo/math/` and related modules that provide core geometric computations.

## Purity & Layering

1. Keep modules free of side effects: no file I/O, logging, or global state. Accept all tensors on the caller’s device and return tensors directly.
2. Avoid imports from higher layers (`datasets`, `models`, `_cpp`). The math layer is the foundation; higher layers may depend on it, never the reverse.
3. Preserve deterministic behavior. Take `torch.Generator` handles when randomness is unavoidable and leave seeding decisions to callers.

## Symplectic Geometry Focus

- Primary research target: 4D polytopes in the symplectic standard setting. Functions should expose interfaces that generalize cleanly to additional dimensions.
- Use semantic tensor names (`normals`, `offsets`, `support_values`). Document coordinates, units, and orientation expectations.
- When introducing new geometric primitives, include short derivations or invariants in docstrings to aid reviewers and future agents.

## Torch Integration

- Default dtype to `torch.float64` unless an existing API states otherwise. Never downcast silently; surface dtype arguments if multiple precisions are required.
- Do not move tensors across devices implicitly. Validate inputs (`assert tensor.device == reference.device`) only when it protects against known bugs.
- Leverage vectorized operations whenever feasible; prefer `torch.einsum`, `torch.linalg`, and broadcasting over Python loops.
- Push plotting/IO conversions to call sites (e.g., `tensor.detach().cpu().numpy()`); keep math functions returning tensors.

## Naming & Structure

- Modules should group related concepts (e.g., `support.py`, `convex.py`) rather than accumulating miscellaneous helpers.
- Exported functions should have docstrings in Google style, highlighting tensor shapes.

### Minimal Example (PyTorch)
```python
import torch

def support(points, direction):
    """Support function of a finite point set.

    Args:
      points: (N, D) float64
      direction: (D,) float64, not normalized

    Returns:
      () float64 tensor:
        max_i <points[i], direction>
    """
    return (points @ direction).max()
```
- Keep module-level comments minimal; prefer coverage in docstrings or `docs/` when detailed exposition is necessary.

## Testing & Validation

- Favor property-based or invariance tests (e.g., support functions respect rotation or translation). Use fixed `torch.Generator` instances for repeatability.
- Put smoke-tier tests under `tests/math/` or similar module-aligned paths. For heavier property sweeps, coordinate with maintainers before adding long-running suites.
- Integrate precision checks (`torch.testing.assert_close`) with tolerances grounded in theory (e.g., `rtol=1e-9` for double precision).

## Escalation Guidelines

- Escalate when a proposed change alters asymptotic complexity, deviates from established naming, or introduces new C++ dependencies.
- Coordinate with the maintainer before modifying existing `_cpp` bindings or adding new ones.

## Related Skills

- `good-code-loop` — general architectural and style rules.
- `testing-and-ci` — ensures validation accompanies geometry changes.
- `performance-discipline` — governs profiling and optimization decisions if math routines become bottlenecks.
