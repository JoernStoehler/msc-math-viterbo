# Certified C*(X) Construction for Chaidez–Hutchings Pruning (R^4)

Status: Draft (skeleton for spec completion)

Author(s): TBD

## Scope

- Deliver a mathematically certified construction of the constant C*(X) used in the Chaidez–Hutchings (CH) per-face budget for the oriented-edge DFS in R^4.
- C*(X) must satisfy the inequality from the paper (Theorem 1.12(v), Lemma 5.13, §6.2): for any Type‑1 combinatorial orbit γ with 2‑face endpoints F_1,…,F_k and any rotation-number bound R on the approximating smooth orbits,
  
  sum_i C*(X) · θ(F_i) ≤ R,
  
  where θ(F) is the spherical angle between the outward unit normals of the two 3‑faces adjacent to the 2‑face F.
- Out of scope: implementation code; this document is math/spec only. Implementation will follow in a separate task once the spec is reviewed and approved.

## Acceptance Criteria (Spec)

- Defines all symbols and assumptions precisely (symplectic polytope in R^4; outward unit facet normals; 2‑faces non‑Lagrangian).
- States and cites the exact CH inequalities used (Theorem 1.12(v), Lemma 5.13, §6.2) and explains how C*(X) slots into the budget.
- Provides a rigorously justified construction of C*(X) using explicit, computable uniform lower bounds that depend only on X (H‑representation and 2‑face bases):
  - A global positive lower bound on the “normal component” |(i v)_N| across the strata relevant to the smoothing (0/1/2‑faces), as in §5.
  - A global positive lower bound on the denominators appearing in the smoothing’s v′(s) formula (derived from convexity and 0 ∈ int(X)).
  - A lower bound on the S^3 path length across 2‑face transitions: dist_S^3(ν_E, ν_E′) = θ(F) (§6.2, Step 2).
  - A clearly written composition of these bounds into a single constant C*(X) with units consistent with the rotation-number budget.
- Specifies concrete, numerically computable expressions (closed form or small convex subproblems) for each uniform lower bound and for C*(X), including numerical stability considerations (e.g., conditioning, degeneracy checks, tolerances).
- Lists invariance/monotonicity properties C*(X) must satisfy (e.g., translation invariance, appropriate scaling behavior) and shows the construction respects them.
- Includes a verification section with example classes (e.g., product polytopes) illustrating how the construction behaves and how to validate Σ C*(X)·θ ≤ R in practice, without relying on unproved heuristics.

## Problem Setup and Notation

- X ⊂ R^4: convex “symplectic polytope” (every 2‑face F is non‑Lagrangian).
- Facets (3‑faces): E, with outward unit normals ν_E ∈ S^3 and support numbers c_E > 0 (H‑representation).
- 2‑faces: F between adjacent facets E_+, E_−; basis B_F ∈ R^{4×2} spanning T F; spherical angle θ(F) := arccos⟨ν_{E_+}, ν_{E_−}⟩ ∈ (0, π).
- J ∈ R^{4×4}: standard symplectic form.
- For Type‑1 combinatorial orbit γ with endpoints on 2‑faces {F_i}, CH budget has the form Σ C*(X)·θ(F_i) ≤ R.

## Required Inequalities (from the paper)

1) (S^3 step) Each smoothing arc that crosses from 3‑face E to E′ induces a path in S^3 from ν_E to ν_{E′} with length ≥ dist_S^3(ν_E, ν_{E′}) = θ(F) (Theorem 1.12 proof, §6.2 Step 2).
2) (Rotation vs. speed) Lemma 5.13: rotation number ρ satisfies ρ ≥ C_speed(X) · ∫ |v′(s)| ds for a uniform positive constant C_speed(X) depending only on X.
3) (Composition) Combining (1) and (2) yields Σ C_speed(X)·θ(F_i) ≤ ρ ≤ R.
4) Therefore C*(X) := C_speed(X) is admissible for the CH budget Σ C*(X)·θ(F) ≤ R.

Spec must show how to compute C_speed(X) explicitly.

## Construction Outline for C*(X)

1) Uniform denominator lower bound: derive an explicit positive lower bound D_min(X) for the denominators appearing in the smoothing’s formula for v′(s) (cf. Eq. (5.1) in the paper), across all relevant strata (0/1/2‑faces), leveraging convexity and 0 ∈ int(X). For 2‑faces between facets i,j with offsets c_i,c_j and unit normals ν_i,ν_j, show how to bound
   
   inf_{a,b ≥ 0, not both 0} (a c_i + b c_j) / ||a ν_i + b ν_j|| > 0,
   
   then extend to 1‑ and 0‑faces per §5.

2) Uniform “normal component” lower bound: derive an explicit positive lower bound N_min(X) for |(i v)_N| across the same strata, using the active facet normals and the tangent/normal projectors derived from the face bases (as in §5). Provide formulas or small convex subproblems to compute these bounds robustly from (normals, offsets, B_F).

3) Assemble C_speed(X): combine the bounds to produce a constant C_speed(X) > 0 with correct units and scaling such that ρ ≥ C_speed(X) ∫ |v′(s)| holds (per Lemma 5.13). Provide the exact formula and a short justification of its derivation consistent with the original proof.

4) Define C*(X) := C_speed(X). Then specify c_F := C*(X)·θ(F) and the CH budget Σ_F c_F ≤ R.

## Numerical Computation Details

- Algorithms to compute D_min(X), N_min(X) robustly from H‑rep and face bases, including:
  - Enumeration and handling of 2‑faces (already available in code; spec should be self-contained).
  - Recommended tolerances and conditioning safeguards (e.g., handling near‑collinear ν_i, ν_j; degenerate cones; QR‑based projections from B_F).
  - Convex subproblems: 1D minimization over (a,b)≥0 on rays parameterized by a/b; guarantees for uniqueness/existence on compactified domain.

## Invariance and Scaling

- State and justify the invariance properties of C*(X) under translation, and behavior under uniform scaling of X (ensure compatibility with the scaling of the rotation‑number budget).

## Verification Plan (for the spec)

- Checklist to verify the spec is complete and sound:
  - Every inequality used is either proved or cited verbatim from the paper with precise references.
  - Every bound (D_min, N_min) is an explicit, computable expression from (normals, offsets, B_F) with a correctness argument.
  - The final composition into C*(X) matches the structure and constants used in Lemma 5.13/§6.2.
  - Example walkthroughs on small polytopes (e.g., product polytopes) showing how to compute C*(X) end‑to‑end from H‑rep and face bases.

## References

- Chaidez, J., Hutchings, M. “Computing Reeb dynamics on 4d convex polytopes.” arXiv:2008.10111 (Sections §2, §5, §6; Lemma 5.13; Theorem 1.12(v)).
