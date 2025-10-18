# Certified C*(X) for CH Per-Face Budget (R^4)

Status: Rigorous spec (math only)

## Scope

- Provide a precise, computable specification of the certified constant C*(X) used in the Chaidez–Hutchings per-face budget for pruning in R^4.
- Math/spec only; no implementation in this document.
- Ensure consistency with Chaidez–Hutchings: Theorem 1.12(v), Lemma 5.13, and §6.2 (proof structure: denominator bound, normal-component bound, S^3 path-length across 2-face transitions, then combine).

## Standing Assumptions and Notation

- Ambient space: R^4 with standard symplectic form ω and complex structure J (so J^2 = −I and ω(u,v) = ⟨Ju, v⟩).
- Polytope: X ⊂ R^4 is a rational convex polytope with 0 ∈ int(X).
- H-representation:
  - Facets indexed by i ∈ {1,…,m}.
  - Outward facet normals n_i ∈ Z^4 are primitive (no common integer factor); offsets λ_i ∈ R satisfy X = {x ∈ R^4 : ⟨n_i, x⟩ ≤ λ_i for all i}.
  - Unit-normal normalization: ν_i := n_i/||n_i|| and c_i := λ_i/||n_i||. When convenient, normalize offsets by ĉ_i := c_i / max_j c_j to obtain scale-free quantities.
- Faces and active sets:
  - A 2-face F is the nonempty intersection of exactly two supporting facets i,j with all other inequalities strict near F.
  - A 1-face (edge) is the nonempty intersection of exactly three facets i,j,k; a 0-face (vertex) is the nonempty intersection of exactly four facets i,j,k,ℓ.
  - For a face with active set I = {i1,…,ik}, define the normal subspace N_I := span_R{ν_{ir}} and the tangent subspace T_I := N_I^⊥ (Euclidean orthogonal).
  - 2-face basis B_F ∈ Z^{4×2}: columns form a primitive lattice basis of the rank-2 lattice L_F := Z^4 ∩ T_{ {i,j} }. Any such primitive basis is admissible; numerical work should use an orthonormalization (QR) of B_F to form a tangent frame.
- Spherical angles:
  - For a 2-face F bounded by facets i and j, define θ(F) := arccos⟨ν_i, ν_j⟩ ∈ (0, π).
  - For unit vectors u, v ∈ S^3, dist_S^3(u,v) := arccos⟨u, v⟩.

Remark on “symplectic polytope.” The only property used here is that along the open 1-face and 0-face strata occurring in smoothing transitions (cf. §6.2), the geometric quantities below are strictly positive. Interior 2-face sheets can be J-invariant (hence |(Jv)_N| may vanish there), but transitions occur only across 1/0-face neighborhoods in the smoothing considered by CH.

## CH Budget Context and Required Inequalities

- Theorem 1.12(v) and §6.2: As one passes across a 2-face F between adjacent facets i and j, the smoothed direction field v(s) traces a path in S^3 whose length is at least dist_S^3(ν_i, ν_j) = θ(F).
- Lemma 5.13 (speed/rotation lower bound): There exists C_speed(X) > 0, depending only on X, such that for any smoothed Reeb segment parameterized by s,
  
  ρ ≥ C_speed(X) · ∫ |v′(s)| ds.
  
  Here ρ is the rotation number contribution of the segment (dimensionless), and v(s) ∈ S^3.
- Combining these yields, for any Type‑1 combinatorial orbit γ with 2-face crossings F_1,…,F_k and any admissible smoothed realization with rotation bound R,
  
  Σ_{r=1}^k C_speed(X) · θ(F_r) ≤ ρ ≤ R.
  
  Hence any C*(X) ≤ C_speed(X) is admissible; we specify C*(X) := C_speed(X).

The rest of the spec makes C_speed(X) explicit and computable from the H-representation and per-2-face bases.

## Denominator Bounds: D_min(X) and U_max(X)

Link to CH. Along a smoothed segment with y ∈ ∂X, unit v ∈ N_y^+X and small ε>0, equation (s5, Eq. (Reebsmoothed) specialized in Lemma 5.13) gives

- v′(s) = 2 ε^{-1} (i v)_N / (⟨v,y⟩ + ε).

Thus the key “denominator” is ⟨v,y⟩ + ε. We need:
- A uniform positive lower bound D_min(X) on ⟨v,y⟩ to control |v′| from above (used in Lemma 5.13 to pass from ε^{-1}(b−a) to ∫|v′|).
- A uniform finite upper bound U_max(X) on ⟨v,y⟩ to control r_ε^{min} from below in Lemma 5.13 (cf. s5, Eq. (remin)).

Setup per active set. Let I={i1,…,ik} (k∈{2,3,4}) be the active facets at y. Let N_I be the k×4 matrix with rows ν_{ir}^T, and let c_I ∈ R^k collect the offsets c_{ir}. Any unit v ∈ N_y^+X can be written as

- v = N_I^T w / ||N_I^T w|| for some w ∈ R_+^k \ {0}.

At such y, the support identity gives

- ⟨v,y⟩ = (w·c_I) / ||N_I^T w||.

Exact convex formulations.
- D(I) := inf_{w≥0, w≠0} (w·c_I)/||N_I^T w|| = min_{w≥0, ||N_I^T w||=1} w·c_I.
  - SOCP: minimize w·c_I subject to ||N_I^T w||_2 ≥ 1 and w≥0. Optimum attains ||N_I^T w||_2=1.
- U(I) := sup_{w≥0, w≠0} (w·c_I)/||N_I^T w|| = max_{w≥0, ||N_I^T w||≤1} w·c_I.
  - SOCP: maximize w·c_I subject to ||N_I^T w||_2 ≤ 1 and w≥0. Optimum attains ||N_I^T w||_2=1.

Global bounds.
- D_min(X) := min over active I of D(I).
- U_max(X) := max over active I of U(I).

Closed form for k=2. If I={i,j}, with angle θ=arccos⟨ν_i,ν_j⟩∈(0,π), then
- D(I) = min_{t≥0} (c_i + t c_j)/||ν_i + t ν_j||.
- U(I) = max_{t≥0} (c_i + t c_j)/||ν_i + t ν_j||.
The minimizer t* (when finite) solves (c_j cos θ − c_i) t + (c_j − c_i cos θ) = 0, i.e.
- t* = (c_i cos θ − c_j)/(c_j cos θ − c_i) if t*≥0, else attained at t∈{0,∞} giving min{c_i,c_j} or max{c_i,c_j} respectively. Evaluate both D(I) and U(I) from these candidates.

Coarse certified bounds (optional, easy to compute).
- D(I) ≥ (min_r c_{ir})/σ_max(N_I) and U(I) ≤ (max_r c_{ir}) √k / σ_min(N_I), using ||N_I^T w||_2 ∈ [σ_min||w||_2, σ_max||w||_2] and ||w||_1 ≤ √k||w||_2.

Remarks.
- D_min and U_max are translation-invariant under 0∈int(X) normalization and scale linearly with X↦sX (both multiply by s). They are finite and strictly positive by compactness and strict convex support at boundary.

## Quaternionic Normal-Component Bound N_ann(X)

Link to CH. Lemma 5.13 uses the inequality (s5, Eq. (annest))

- |(i v)_N|^2 + |(cos θ j v + sin θ k v)_N|^2 ≥ C0(X)

uniformly over all relevant boundary points y, unit v ∈ N_y^+X not normal to a 3-face adjacent to the stratum, and all θ∈R/2πZ. Here i is the standard complex structure, and j,k complete an orthonormal quaternionic triple as in §4 of CH.

Computable lower bound per active set. Fix I with k∈{2,3,4}. Let R_I ∈ R^{4×k} have orthonormal columns spanning N_I. Define the k×k blocks

- A_I := R_I^T i N_I^T,
- B_I := R_I^T j N_I^T,
- C_I := R_I^T k N_I^T,
- H_I := N_I N_I^T (SPD).

For θ∈[0,2π), set M_I(θ) := A_I^T A_I + (cos θ B_I + sin θ C_I)^T(cos θ B_I + sin θ C_I). Then for any w∈R^k,

- ||A_I w||^2 + ||(cos θ B_I + sin θ C_I) w||^2 = w^T M_I(θ) w.

Dropping the nonnegativity constraint on w gives the certified bound

- N_ann(I) := min_{θ∈[0,2π)} λ_min(H_I^{-1/2} M_I(θ) H_I^{-1/2}).

Global bound.
- N_ann(X) := min over active I of N_ann(I) > 0.

Computation notes.
- N_ann(I) reduces to a 1D minimization of the smallest eigenvalue of a k×k symmetric matrix-valued trigonometric polynomial. Use adaptive sampling with a Lipschitz bound L_I ≥ sup_θ ||d/dθ(H_I^{-1/2} M_I(θ) H_I^{-1/2})||_2 to certify a lower bound to tolerance.
- A convenient derivative formula is
  M_I(θ) = A_I^T A_I + B_I^T B_I cos^2θ + C_I^T C_I sin^2θ + (B_I^T C_I + C_I^T B_I) sinθ cosθ.
- The bound is invariant under replacing (j,k) by any orthonormal basis of the 2-plane orthogonal to i; any fixed canonical choice (e.g., CH §4 quaternionic matrices) is acceptable in implementation.

## S^3 Path-Length Lower Bound at 2-Face Transitions

For a 2-face F with adjacent facets i and j, the smoothed direction v(s) must traverse at least the spherical distance between the two outward unit facet normals (§6.2):

- L(F) ≥ dist_S^3(ν_i, ν_j) = θ(F) = arccos⟨ν_i, ν_j⟩.

This bound is independent of offsets and depends only on directions of adjacent facet normals.

## Putting It Together: C_speed(X) and C*(X)

Two CH ingredients (s5).
- From Lemma 5.13, using equations (remin) and (svepsilon) and the annest bound, one gets a lower bound on the rotation number of any smoothed segment avoiding 3-face smoothings:
  ρ ≥ [N_ann(X)/(π (U_max(X)+ε))] · ε^{-1} · (b − a).
- Also, from v′(s) = 2 ε^{-1} (i v)_N / (⟨v,y⟩ + ε) and |(i v)_N| ≤ 1 and ⟨v,y⟩ ≥ D_min(X),
  ∫_a^b |v′(s)| ds ≤ [2/(D_min(X)+ε)] · ε^{-1} · (b − a).

Divide the two inequalities to obtain, for all sufficiently small ε>0,
- ρ ≥ [ N_ann(X) · (D_min(X)+ε) / (2 π (U_max(X)+ε)) ] · ∫_a^b |v′(s)| ds.

Letting ε→0, define the certified speed constant

- C_speed(X) := [ N_ann(X) · D_min(X) ] / [ 2 π · U_max(X) ].

S^3 lengths and per-face budget. Section §6.2 shows each 2-face crossing contributes at least θ(F) = dist_S^3(ν_i,ν_j) to the S^3 path length. Summing over crossings in a Type‑1 combinatorial orbit γ gives L_S^3(γ) ≥ Σ_F θ(F). Combining with Lemma 5.13,

- ρ ≥ C_speed(X) · L_S^3(γ) ≥ C_speed(X) · Σ_F θ(F).

We set

- C*(X) := C_speed(X), and per-face weights c_F := C*(X) · θ(F), so Σ_F c_F ≤ ρ.

## Explicit Algorithm (H-Rep + Face Bases)

Inputs.
- Primitive facet normals n_i ∈ Z^4 and offsets λ_i ∈ R.
- Unit normals ν_i = n_i/||n_i||.
- For each 2-face F with facets {i,j}, any primitive tangent basis B_F of L_F (used only for diagnostics; the formulas below use ν_i directly).

Steps.
1) Enumerate active sets I that occur (pairs for 2‑faces, triples for 1‑faces, quadruples for vertices) by feasibility of equalities ⟨n_i, x⟩ = λ_i.
2) For each I, compute D(I) by the SOCP: minimize w·c_I subject to ||N_I^T w||_2 ≥ 1 and w≥0. Set D_min(X) = min_I D(I).
3) For each I, compute U(I) by the SOCP: maximize w·c_I subject to ||N_I^T w||_2 ≤ 1 and w≥0. Set U_max(X) = max_I U(I).
4) For each I, build an orthonormal basis R_I for N_I, and assemble A_I, B_I, C_I and H_I. Compute N_ann(I) = min_{θ∈[0,2π)} λ_min(H_I^{-1/2} M_I(θ) H_I^{-1/2}) via adaptive 1D search with certified Lipschitz bounds; set N_ann(X) = min_I N_ann(I).
5) Set C*(X) = [N_ann(X) · D_min(X)] / [2 π · U_max(X)].
6) For each 2‑face F bounded by facets (i,j), compute θ(F) = arccos⟨ν_i, ν_j⟩ and per‑face budget c_F := C*(X)·θ(F).

Conditioning and certification.
- Use QR/SVD throughout; avoid forming inverses. For k=2, the closed‑form formulas for D(I), U(I) serve as checks.
- If σ_min(N_I) is small (nearly dependent normals), D(I) shrinks and U(I) grows, shrinking C*(X). This reflects genuine geometric degeneracy and is expected.
- Integer fallback: lower bound σ_min(N_I) using determinants of minors and norm bounds to get coarse certified bounds for D(I) and U(I).
- N_ann(I) Lipschitz bound: using M_I′(θ) derived above and ||H_I^{-1/2}·|| operator norm, take L_I ≥ ||H_I^{-1/2}||^2 · sup_θ||M_I′(θ)||_2. Adaptive sampling with interval certificates then yields a rigorous lower bound.

## Invariance and Scaling

- Translation invariance: With 0 ∈ int(X), ⟨v,y⟩ depends only on offsets relative to the origin; translating X while keeping 0 in the interior is normalized away in the standard CH setup. The constants D_min, U_max, N_ann, hence C*(X), are translation-invariant under this normalization.
- Rotation invariance: For any orthogonal U ∈ SO(4), replacing X by U X and J by U J U^T leaves D_min, N_min, and C*(X) unchanged. In particular, unitary U ∈ U(2) preserves J and keeps C*(X) invariant.
- Scale invariance: Using unit normals ν_i makes D_min and N_min dimensionless; thus C*(X) is invariant under uniform scaling X ↦ sX. The per-face angles θ(F) are also scale-invariant.
- Monotonicity under coarsening: Refining the polytope by adding facets can only decrease D_min (or leave it unchanged), thereby not increasing C*(X). This matches the intuition that more acute normal cones permit slower transitions.

## Correctness and Safety (Proof Sketch)

- From s5 (Eq. (remin)) and S(V)=ε^{-1}|V_N|^2 (Eq. (svepsilon)), r_ε^{min}(y+εv) ≥ [ε^{-1}/(π(⟨v,y⟩+ε))] · C0(X), where C0(X) is any uniform positive lower bound for the annest expression. By construction, C0(X) ≥ N_ann(X).
- Using U_max(X) ≥ sup ⟨v,y⟩ and the above, integrate to obtain ρ ≥ [N_ann(X)/(π(U_max(X)+ε))] ε^{-1}(b-a).
- From v′(s) = 2 ε^{-1} (i v)_N/(⟨v,y⟩+ε) and |(i v)_N| ≤ 1 and ⟨v,y⟩ ≥ D_min(X), we get ∫|v′| ≤ [2/(D_min(X)+ε)] ε^{-1}(b-a).
- Combining gives ρ ≥ [N_ann(X)(D_min(X)+ε)/(2π(U_max(X)+ε))] ∫|v′|. Letting ε→0 yields ρ ≥ C_speed(X) ∫|v′| with C_speed(X) = [N_ann·D_min]/[2π U_max].
- Section §6.2 shows the S^3 path between adjacent facets has length ≥ θ(F), and these lengths add along a Type‑1 orbit. Hence ρ ≥ C_speed(X) Σ_F θ(F).
- Therefore setting C*(X)=C_speed(X) is safe: any pruning that enforces Σ_F C*(X)·θ(F) ≤ R cannot eliminate a true minimal cycle with rotation-number budget ≤R.

## Verification Checklist

- Face enumeration correct: active sets I = {pairs, triples, quadruples} correspond to actual faces (feasibility of equalities, strictness of others).
- D_min computed by SOCP per I (or pairwise closed form), then minimized across I.
- U_max computed by SOCP per I (or pairwise closed form), then maximized across I.
- N_ann computed as min_θ λ_min(H_I^{-1/2} M_I(θ) H_I^{-1/2}) per I, then minimized across I; numerical procedure includes a certificate of the lower bound.
- Per-face angles θ(F) computed as arccos⟨ν_i, ν_j⟩ for the two facets bounding F.
- Final budget constant C*(X) = D_min · N_min and per-face weights c_F = C*(X)·θ(F) satisfy Σ_F c_F ≤ R in any test case consistent with Theorem 1.12(v) and Lemma 5.13.
- Numerical stability: QR/SVD routines converge; tolerances chosen so that near-degenerate configurations are flagged.

## Small Examples

Example 1 (Cross-polytope/cube normals; 0 ∈ int). Let the outward unit facet normals be the eight vectors ±e_1, …, ±e_4 (active sets use orthogonal normals).
- For any 2-face bounded by e_a and e_b with a≠b, θ(F) = arccos⟨e_a, e_b⟩ = π/2.
- D(I)=U(I)=1 for all I (pairwise closed form). Hence D_min=U_max=1.
- For every I, H_I=I and A_I,B_I,C_I are orthogonal blocks; thus N_ann(I)=1 and N_ann(X)=1.
- Therefore C*(X) = 1/(2π)·2π? No: C*(X) = [N_ann·D_min]/[2π U_max] = 1/(2π)·? With D_min=U_max=N_ann=1, C*(X)=1/(2π). Each 2-face contributes θ=π/2, so per-face budget is π/(4π)=1/4; summing matches the CH scaling after trivialization conventions.

Example 2 (Two non‑orthogonal facets). Let a 2‑face be bounded by unit normals ν_i, ν_j with mutual angle θ∈(0,π), offsets c_i,c_j>0.
- D({i,j}) = min_{t≥0} (c_i + t c_j)/||ν_i + t ν_j|| (closed form via t* above). U({i,j}) uses the same expression with max.
- For any active triple including {i,j}, assemble A_I,B_I,C_I and compute N_ann(I); the global N_ann(X) is the minimum across all relevant I.
- As θ→0 (nearly parallel facets), D({i,j})→min(c_i,c_j) while U({i,j})→max(c_i,c_j), and N_ann(I) may decrease if a bad 1‑face occurs; consequently C*(X) shrinks, reflecting slower certified rotation growth through a nearly flat transition.

## References

- Chaidez, J.; Hutchings, M. Computing Reeb dynamics on 4d convex polytopes. arXiv:2008.10111. See Lemma 5.13, §6.2, and Theorem 1.12(v).

## Quaternionic Conventions (for implementation)

- Identify R^4 with the quaternions H via the standard basis, and fix the quaternionic units acting by left multiplication. One explicit choice of 4×4 matrices is:
  - i = [[0,−1,0,0],[1,0,0,0],[0,0,0,−1],[0,0,1,0]] (this is the standard complex structure J),
  - j = [[0,0,−1,0],[0,0,0,−1],[1,0,0,0],[0,1,0,0]],
  - k = i·j = [[0,0,0,−1],[0,0,1,0],[0,−1,0,0],[1,0,0,0]].
- Any orthonormal quaternionic triple unitarily equivalent to the above yields the same certified constants (N_ann, hence C*(X)) due to the invariance noted earlier.
