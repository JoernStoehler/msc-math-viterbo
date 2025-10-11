status: in-progress
created: 2025-10-11
workflow: task
summary: Select and implement a modern EHZ spectrum algorithm (no legacy deps).
---

# Task: Modern EHZ Spectrum Algorithm

Goal: Implement `viterbo.modern.spectrum` without dependencies on legacy modules, define padding semantics, and add smoke/deep tests.

Context
- Modern capacity and 4D cycles are implemented and decoupled from `exp1`/`symplectic`.
- The spectrum API remains blocked. Tests currently xfail by design.
- Padding choice: use in-band invalidation only — `NaN` (floats) and `-1` (indices). No extra mask arrays.

Candidate approaches
- Oriented-edge cycles (4D):
  - Enumerate simple directed cycles over the oriented-edge graph; compute actions per cycle using polytope geometry (e.g., facet-normal Reeb measures or piecewise-linear Minkowski length when available). Sort and return top-`head`.
  - Pros: aligns with our 4D cycle infra; natural for products and many polytopes.
  - Cons: 4D only without generalization.

- Facet-subset stationary measures:
  - Reuse the Haim–Kislev per-subset Reeb measures; derive action values across admissible facet orders, yielding a discrete spectrum. Return top-`head` over all subsets.
  - Pros: general `2n`-dimensional handle; shares structure with capacity reference.
  - Cons: combinatorial; requires careful pruning and numeric stability.

- Billiard-based discrete actions (products):
  - For `K×T` products, compute actions of admissible billiard words using normal fan or direct word enumeration; return top-`head`.
  - Pros: fast and interpretable for product families.
  - Cons: limited scope; needs fallback for generic polytopes.

Recommended plan (phased)
1) 4D-oriented-edge baseline (DONE)
   - Implemented cycle enumeration bounded by `head` and Euclidean PL action; returns sorted `head`.
   - NaN padding semantics defined for batched API.
2) General `2n`: facet-subset extension (NEXT)
   - Compute candidate actions from admissible subsets using the measure system as capacity; prune with DP-based maxima; return `head`.
3) Optional product specialization (LATER)
   - Add fast path for `K×T` via normal-fan enumeration for improved coverage.

API and semantics
- `ehz_spectrum_reference(polytope, head)` → list[float]
  - Returns the smallest `head` actions (ascending). If fewer actions exist, return a shorter list (reference API).
- `ehz_spectrum_batched(normals, offsets, head)` → Float[Array, " batch head"]
  - Padding with `NaN` for missing entries; no separate mask.

Tests (smoke/deep)
- 4D product square: non-empty spectrum; monotone with `head`; `NaN` padding behavior for batched API. (DONE)
- Add symmetry/scaling checks after action definition stabilizes (LATER).

Risks and mitigations
- Combinatorial blowup: bound cycle length/order, prune via DP shortcuts.
- Numerical stability in measures: keep x64; small Tikhonov regularization where systems are near-singular; deduplicate actions within `rtol, atol`.

Outcome
- A self-contained modern spectrum implementation with clear padding semantics and tests; 4D baseline in place; generalization pending.
