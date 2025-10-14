---
title: minimal_action — EHZ capacities and cycles
---

# `viterbo.math.minimal_action`

Capacity solvers and minimal-action diagnostics with a 4D focus. Current
implementations cover planar (2D) smoke tests; higher-dimensional backends are
stubbed with references to planned algorithms.

## Functions (implemented placeholders)

- `capacity_ehz_algorithm1(normals: (F, d), offsets: (F,)) -> ()` *(2D)*
- `capacity_ehz_algorithm2(vertices: (M, d)) -> ()` *(2D)*
- `capacity_ehz_primal_dual(vertices, normals, offsets) -> ()` *(2D consistency check)*
- `minimal_action_cycle(vertices, normals, offsets) -> (capacity, cycle)` *(2D)*
- `minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p, *, max_bounces=3)
  -> (capacity, cycle)` *(4D Lagrangian product, vertex-contact search)*
- `systolic_ratio(volume: (), capacity_ehz: (), symplectic_dimension: int) -> ()`

## Stubs / planned work

- `minimal_action_cycle_lagrangian_product_generic(normals_q, offsets_q, normals_p, offsets_p)` —
  facet-interior refinement of the Lagrangian product solver
- `capacity_ehz_haim_kislev(normals, offsets)` — general Haim–Kislev formula in 4D+
- `oriented_edge_spectrum_4d(vertices, normals, offsets, *, k_max)` — Hutchings-style spectrum
- `capacity_ehz_via_qp(normals, offsets)` — facet-multiplier convex QP
- `capacity_ehz_via_lp(normals, offsets)` — LP/SOCP relaxations for bounds

## Minimal-action Minkowski billiards on Lagrangian products (4D)

The helper `minimal_action_cycle_lagrangian_product` implements the vertex-contact
search guaranteed by Rudolf’s three-bounce theorem for convex Lagrangian products
in four dimensions.  Let

- `Q ⊂ ℝ²` be a convex polygon describing the ``q``-factor, given by vertices
  `vertices_q = (q₀,…,q_{M−1})` in arbitrary order, and
- `T ⊂ ℝ²` be the dual polygon describing the ``p``-factor through supporting
  halfspaces `normals_p ⋅ p ≤ offsets_p`.

The algorithm returns the minimal Ekeland–Hofer–Zehnder (EHZ) action among closed
Minkowski billiard trajectories on `Q × T` that alternate between vertices of `Q`
and supporting points on facets of `T`.

### Mathematical specification

1. **Input validation.**  Require both polygons to live in the plane and `offsets_p`
   to be strictly positive.  Order the vertices of `Q` counter-clockwise and
   reconstruct the vertex set of `T` from its half-space representation.
2. **Candidate enumeration.**  By Rudolf (2022, Thm. 1.1) a minimal-action closed
   characteristic on `Q × T` can be realised by a broken geodesic with at most
   three bounces on `Q`.  Enumerate every unordered pair and triple of vertices of
   the ordered polygon `Q` and treat them as potential bounce indices.
3. **Support evaluation.**  For each oriented edge `q_j − q_i` in the candidate,
   evaluate the support function of `T` to obtain the unique vertex `p` of `T`
   that maximises `⟨p, q_j − q_i⟩`.  This value equals the Minkowski length of the
   edge with respect to `T^∘` and contributes to the total action.
4. **Reflection constraints.**  Accept only candidates that satisfy the strong
   reflection rules:
   - in the two-bounce case, the segment difference `p_forward − p_backward`
     must expose the chosen vertices of `Q` on both sides;
   - in the three-bounce case, each vertex `q_i` must maximise
     `⟨q_i, −(p_{i} − p_{i−1})⟩` in `Q` (indices modulo the cycle).
5. **Action minimisation.**  Sum the support values attached to the candidate’s
   edges to obtain its action.  Select the candidate with the minimal action,
   breaking ties lexicographically in `(q, p)` coordinates for determinism.
6. **Output.**  Return the minimal action together with the alternating sequence
   `(q₀, p₀, q₁, p₁, …)` that traces the Minkowski billiard once (without
   repeating the initial point).

The helper tolerates floating-point degeneracies by applying a relative tolerance
`tol ≍ √ε` (machine epsilon of the input dtype) to the support and reflection
tests.  It preserves the dtype and device of `vertices_q` in the returned action
and cycle.

### Complexity and limitations

- Enumerating all unordered triples has cubic cost in the number of vertices of
  `Q`; this is practical for the moderate vertex counts arising in symplectic
  experiments (`M ≤ 50`).
- The method assumes the minimiser touches vertices of both polygons; this holds
  for centrally symmetric products and the pentagon counterexample but fails for
  generic Lagrangian products where bounce points slide along facets.  The stub
  `minimal_action_cycle_lagrangian_product_generic` will lift this restriction by
  solving for facet barycentric coordinates.
- The action returned equals the EHZ capacity of `Q × T` by the Minkowski billiard
  characterisation.  It provides a cycle witness that can be fed to downstream
  diagnostics or used as a warm start for facet-interior refinements.

## Notes
## Haim–Kislev facet programme (`capacity_ehz_haim_kislev`)

### Purpose

Return the Ekeland–Hofer–Zehnder (EHZ) capacity of a convex polytope given by
its half-space representation. The routine implements the Haim–Kislev
optimisation principle specialised to four-dimensional phase space ($2n = 4$).

### Inputs

- `normals`: `(F, 4)` tensor whose rows are outward primitive normals
  $b_i \in \mathbb{R}^4$ of the supporting half-spaces.
- `offsets`: `(F,)` tensor with strictly positive support numbers
  $c_i = h_K(b_i)$ such that the polytope is $P = \{x \mid Bx \leq c\}$.

### Output

- Scalar tensor with the EHZ capacity $c_{\mathrm{EHZ}}(P)$.

### Preliminaries

1. Assemble $B \in \mathbb{R}^{F \times 4}$ by stacking `normals` and
   $c \in \mathbb{R}^F$ from `offsets`. The rows of $B$ are denoted
   $b_1,\dots,b_F$ and define $P(B,c)$ as in the Haim–Kislev formula.【F:docs/papers/2024-vallentin-ehz-np-hard/main.tex†L121-L150】
2. Fix the standard symplectic form $\omega(x,y) = x^{\mathsf{T}} J y$ with
   the block matrix $J = \begin{bmatrix} 0 & I_2 \\ -I_2 & 0 \end{bmatrix}$.【F:docs/papers/2024-vallentin-ehz-np-hard/main.tex†L135-L147】
3. The optimisation variable is a non-negative weight vector
   $\beta \in \mathbb{R}^F_{\ge 0}$ satisfying the affine constraints
   $\beta^{\mathsf{T}} c = 1$ (scale normalisation) and
   $\beta^{\mathsf{T}} B = 0$ (force balance on the normals).【F:docs/papers/2024-vallentin-ehz-np-hard/main.tex†L135-L139】

The EHZ capacity is
\[
  c_{\mathrm{EHZ}}(P) = \frac{1}{2}\left(\max_{\sigma,\,\beta}\;\sum_{1 \le j < i \le F}
  \beta_{\sigma(i)} \beta_{\sigma(j)} \, \omega(b_{\sigma(i)}, b_{\sigma(j)})\right)^{-1},
\]
with the maximum taken over permutations $\sigma$ of the facet indices and
feasible weights $\beta$.【F:docs/papers/2024-vallentin-ehz-np-hard/main.tex†L135-L139】

### Specialisation to $\mathbb{R}^4$

The constraints $\beta^{\mathsf{T}} B = 0$ add four linear equations, so every
extreme ray of the feasible cone uses at most five facets; this allows an
explicit enumeration strategy in four dimensions.

1. **Candidate supports.** Enumerate subsets $S \subseteq \{1,\dots,F\}$ with
   $|S| \le 5$ whose normals span $\mathbb{R}^4$. For each set, compute a basis
   for the nullspace of $B_S^{\mathsf{T}}$. When the nullspace is one-dimensional,
   rescale the generator so that $\beta_S \ge 0$ and
   $\beta_S^{\mathsf{T}} c_S = 1$. Discard supports that cannot be scaled to a
   non-negative vector or that violate the normalisation.
2. **Permutation sweep.** For each feasible support $S$, form the skew-symmetric
   matrix $W_S = B_S J B_S^{\mathsf{T}}$. Evaluate the objective for all
   permutations $\sigma$ of the active indices via
   \(Q(\sigma) = \sum_{j<i} \beta_{\sigma(i)} \beta_{\sigma(j)} (W_S)_{\sigma(i),\sigma(j)}\).
   Since $|S| \le 5$, brute-force evaluation of the $|S|!$ permutations is
   tractable. Ignore permutations with $Q(\sigma) \le 0$ because they do not
   contribute to the maximisation.
3. **Global maximiser.** Track the largest positive value $Q_{\max}$ over all
   supports and permutations. The EHZ capacity is finally
   $c_{\mathrm{EHZ}}(P) = 1 / (2 Q_{\max})$. This matches the quadratic form
   maximisation perspective obtained by permuting the lower-triangular part of
   $B J B^{\mathsf{T}}$.【F:docs/papers/2024-vallentin-ehz-np-hard/main.tex†L315-L375】

### Implementation remarks

- Offsets must remain strictly positive; if the polytope is translated so that
  some $c_i$ vanishes, translate back into the interior before invoking the
  algorithm.
- When multiple supports yield the same $Q_{\max}$, prefer the one with minimal
  cardinality to stabilise the recovered Reeb polygon used by
  `minimal_action_cycle`.
- For degenerate supports where the nullspace has dimension greater than one,
  sample extreme rays (e.g., via linear programming) and treat each candidate as
  above; degeneracy indicates redundant facets or families of Minkowski billiard
  trajectories.
- The brute-force sweep scales combinatorially with $F$, but remains practical
  in the current 4D focus where counterexamples involve fewer than a dozen
  facets. Larger instances can be delegated to a mixed-integer solver that
  implements the same objective with ordering variables.

Notes

- Inputs must satisfy the even-dimension policy (`d = 2n`).
- Implementations preserve dtype/device and rely on helpers from
  `viterbo.math.polytope` and `viterbo.math.constructions`.
