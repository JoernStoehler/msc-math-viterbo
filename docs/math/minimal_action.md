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

- Inputs must satisfy the even-dimension policy (`d = 2n`).
- Implementations preserve dtype/device and rely on helpers from
  `viterbo.math.polytope` and `viterbo.math.constructions`.
