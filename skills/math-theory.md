---
name: math-theory
description: Use for invariants, conventions, and references that guide math implementation and review.
last-updated: 2025-10-17
---

# Math Theory

## Instructions
- Anchor implementation choices in documented invariants and conventions here.
- Cross-check with `docs/math/` for derivations and context; keep code docstrings focused and refer back when needed.

## Focus & Conventions

- Symplectic setting (primarily 4D); APIs should generalize across dimensions where reasonable.
- Semantic names: `normals`, `offsets`, `support_values`, with documented coordinates and orientations.
- Determinism for theoretical checks where applicable; document tolerances grounded in theory.

## Invariants to Check (Examples)

- Support function properties under rotation/translation.
- Volume/capacity relationships under admissible transforms.
- Consistency of polytope representations and hull properties.

## References

- docs/math/index.md — entry point into topic overviews.
- tests/math/* — smoke and property tests illustrating expected invariants.

## Related Skills

- `math-layer` — how to implement these concepts in tensor code.
- `good-code-loop` — ensuring tests and architecture stay aligned with theory.

