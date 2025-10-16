---
title: capacity_ehz — EHZ capacities and cycles
---

# `viterbo.math.capacity_ehz.*`

Capacity solvers and minimal-action diagnostics focused on planar inputs and
Lagrangian products in 4D. The public API is split across small modules:

- `viterbo.math.capacity_ehz.algorithms`
  - `capacity_ehz_algorithm1(normals, offsets)`
  - `capacity_ehz_algorithm2(vertices)`
  - `capacity_ehz_primal_dual(vertices, normals, offsets)`
- `viterbo.math.capacity_ehz.cycle`
  - `minimal_action_cycle(vertices, normals, offsets)`
- `viterbo.math.capacity_ehz.lagrangian_product`
  - `minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p, *, max_bounces=3)`
- `viterbo.math.capacity_ehz.ratios`
  - `systolic_ratio(volume, capacity_ehz, symplectic_dimension)`
- `viterbo.math.capacity_ehz.stubs` (planned backends)
  - `capacity_ehz_haim_kislev(...)`, `capacity_ehz_via_qp(...)`,
    `capacity_ehz_via_lp(...)`, `oriented_edge_spectrum_4d(...)` (implemented)

## Imports (examples)

```python
from viterbo.math.capacity_ehz.algorithms import capacity_ehz_algorithm1, capacity_ehz_algorithm2
from viterbo.math.capacity_ehz.cycle import minimal_action_cycle
from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product
from viterbo.math.capacity_ehz.ratios import systolic_ratio
```

## Mathematical Definitions (concise)

- Support function. For a convex body `T ⊂ ℝ²`, the support function is
  `h_T(v) = max_{p ∈ T} ⟨p, v⟩`. For a planar polygon given by halfspaces
  `B p ≤ c`, the maximiser is attained at a vertex; for a set of vertices the
  maximiser is one of those vertices.

- Action of a polygonal path w.r.t. `T`. For an ordered polygon `(q_0,…,q_{m-1})`,
  the Minkowski length (action) w.r.t. the polar of `T` is
  `A_T(q) = Σ_j h_T(q_{j+1} − q_j)` with cyclic indexing.

- Vertex reflection rule (strong form). A sequence of `q`-vertices and matching
  `p`-support points is admissible if, at each bounce vertex `q_i`, the difference
  of consecutive `p`’s exposes `q_i`:
  `q_i ∈ argmax_{x ∈ Q} ⟨x, p_{next} − p_{prev}⟩`.

These capture the discrete Minkowski billiard setting specialised to convex
Lagrangian products `Q × T ⊂ ℝ⁴` with planar factors.

## Algorithm: Lagrangian Product (≤3 bounces)

We enumerate all two- and three-bounce candidates on the `q`-polygon (Rudolf, 2022):

1) Input: unordered `vertices_q`, planar halfspaces `(normals_p, offsets_p)`.
2) Order `vertices_q` counter‑clockwise. Recover `vertices_p` from halfspaces.
3) For all unordered pairs `(i, j)` with `i < j`:
   - Let `v = q_j − q_i`; if `‖v‖ ≈ 0`, skip.
   - Compute `h_T(v)` and `h_T(−v)` and their maximisers `p_f`, `p_b`.
   - Accept iff `q_i` maximises `⟨x, p_b − p_f⟩` and `q_j` maximises `⟨x, p_f − p_b⟩`.
   - Action `= h_T(v) + h_T(−v)`; stitch the 4‑segment cycle.
4) For all unordered triples `(i, j, k)` with `i < j < k`:
   - Directions `(q_j−q_i, q_k−q_j, q_i−q_k)` must be non‑degenerate.
   - For each direction, get `h_T(·)` and the maximising vertices `(p_0, p_1, p_2)`.
   - Accept iff the three vertex reflection checks hold with directions
     `−(p_0−p_2)`, `−(p_1−p_0)`, `−(p_2−p_1)` at `q_i`, `q_j`, `q_k`.
   - Action `= Σ h_T(direction)`; stitch the 6‑segment cycle.
5) Return the admissible candidate with minimal action, breaking ties
   lexicographically on the flattened cycle for determinism.

Pseudocode (read‑only outline):

```python
def minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p):
    Q = order_ccw(unique(vertices_q))
    P = halfspaces_to_vertices(normals_p, offsets_p)
    best = None
    # Two-bounce
    for i < j:
        v = Q[j] - Q[i]
        if degenerate(v):
            continue
        a_f, p_f = support_argmax(P, v)
        a_b, p_b = support_argmax(P, -v)
        if a_f <= 0 or a_b <= 0:
            continue
        if not exposes(Q, i, p_b - p_f) or not exposes(Q, j, p_f - p_b):
            continue
        best = update(best, a_f + a_b, cycle_from(i, j, p_f, p_b))
    # Three-bounce
    for i < j < k:
        dirs = (Q[j]-Q[i], Q[k]-Q[j], Q[i]-Q[k])
        if any(degenerate(d) for d in dirs):
            continue
        acts, ps = zip(*[support_argmax(P, d) for d in dirs])
        if any(a <= 0 for a in acts):
            continue
        if not (exposes(Q, i, -(ps[0]-ps[2])) and exposes(Q, j, -(ps[1]-ps[0]))
                and exposes(Q, k, -(ps[2]-ps[1]))):
            continue
        best = update(best, sum(acts), cycle_from(i, j, k, ps))
    if best is None:
        raise ValueError("no admissible trajectory")
    return best
```

## Planar placeholders (2D)

- `capacity_ehz_algorithm1(B, c)` → area of the polygon recovered from `(B, c)`.
- `capacity_ehz_algorithm2(V)` → area of the polygon `V`.
- `minimal_action_cycle(V, B, c)` → returns `V` ordered CCW and its area.

These are smoke‑test implementations, not the general EHZ formula in 2D.

## References (selected)

- Artstein‑Avidan, S.; Ostrover, Y. “From symplectic measurements to the
  Mahler conjecture.” (Minkowski billiards and EHZ program)
- Rudolf, M. (2022). “On closed Minkowski billiards in the plane.”
  (Every minimal closed orbit on a convex polygon has ≤3 bounces.)
- Haim‑Kislev, S. (2024). “On the Ekeland–Hofer–Zehnder capacity.”
  (Facet‑multiplier formulations and optimisation principles.)

## Haim–Kislev Facet‑Multiplier Formulation

Let a convex polytope in `ℝ^{2n}` be given by halfspaces `B x ≤ c` with rows
`b_i^⊤` and strictly positive supports `c_i`. For the standard symplectic form
`J`, define the symplectic pairing `ω(u, v) = u^⊤ J v`. Following
Haim–Kislev, the EHZ capacity admits the variational characterisation

\[
  c_{\mathrm{EHZ}}(P)\;=\;\frac{1}{2}\Big(\max_{\sigma,\,\beta}\;\sum_{1 \le j < i \le F}
  \beta_{\sigma(i)} \beta_{\sigma(j)} \, \omega(b_{\sigma(i)}, b_{\sigma(j)})\Big)^{-1},
\]

with the maximum taken over all permutations `σ` of facet indices and over
weights `β ∈ ℝ^F_{≥0}` that satisfy the affine constraints
`β^⊤ c = 1` and `β^⊤ B = 0`. The latter expresses force balance of the
facet normals. The objective can be recognised as a quadratic form in `β` based
on the skew matrix `W = B J B^⊤`. The permutation lets one pick a consistent
triangular part of `W` so the antisymmetry does not cancel.

### 4D extreme supports and finite enumeration

In `ℝ^4` the linear constraints add four equalities; extreme rays of the feasible
cone `\{β ≥ 0 : β^⊤ B = 0\}` therefore have support size at most five. After the
normalisation `β^⊤ c = 1`, every maximiser can be represented with `|supp(β)| ≤ 5`.
Hence, an exact algorithm in 4D is: enumerate all facet subsets `S` with
`|S| ≤ 5` spanning `ℝ^4`, solve the one‑dimensional nullspace of `B_S^⊤` to obtain
`β_S ≥ 0` with `β_S^⊤ c_S = 1`, and evaluate the quadratic objective for all
`|S|!` permutations restricted to `S`. The global maximum determines
`c_{\mathrm{EHZ}}(P)`. Proofs and equivalences appear in Haim–Kislev (and related
expositions); this yields completeness for all convex polytopes in 4D.

### Algorithm: general 4D polytope (exact)

Pseudocode outline for the facet‑enumeration method:

```python
def capacity_ehz_4d_exact(normals: (F,4), offsets: (F,)) -> float:
    B, c = normals, offsets
    best = -inf
    for S in subsets({1..F}, max_size=5):
        if rank(B[S]) < 4:
            continue
        # Nullspace of B_S^T is 1D at extreme points
        z = nullspace(B[S].T)
        if dim(z) != 1:
            # handle degeneracy by enumerating extreme rays of the cone
            for ray in extreme_rays(B[S].T):
                try_support(ray)
            continue
        try_support(z[:,0])

    if best <= 0:
        raise ValueError("no feasible support found")
    return 1.0 / (2.0 * best)

    def try_support(v):
        beta = v / (c[S] @ v)           # scale so beta^T c_S = 1
        if any(beta < -tol):
            return
        beta = clip_min(beta, 0.0)      # guard tiny negatives
        W = B[S] @ J @ B[S].T
        # Evaluate triangular sums via permutations
        for order in permutations(S):
            q = 0.0
            for i in range(1, len(order)):
                for j in range(0, i):
                    q += beta[order[i]] * beta[order[j]] * W[order[i], order[j]]
            best = max(best, q)
```

Complexity is combinatorial in the number of facets `F`, but tractable for
small polytopes (`|S| ≤ 5`, at most `5! = 120` permutations per support).

Correctness in 4D follows from the extreme-support argument and the variational
equivalence; see Haim–Kislev for rigorous proofs and related convex-optimisation
reductions (e.g., QP/SOCP relaxations). Our repository exposes stubs for a QP
path and will align its implementation to this outline.

### Backend implementation status

The function `capacity_ehz_haim_kislev(B, c)` now implements the exact facet
programme. It enumerates all subsets of at most `d + 1` facets (the extreme ray
support bound from Haim–Kislev, Prop. 3.4), solves the one-dimensional nullspace
of `B_S^⊤`, and maximises the triangular part of `W = B J B^⊤` over all
permutations. Numerical tolerances follow `tol = max(√ε, 1e-9)` in the solver’s
dtype and device. The routine raises a `ValueError` if:

- the ambient dimension is odd or fewer than `d + 1` facets are supplied,
- offsets are non-positive (no bounded polytope),
- or no feasible non-negative multiplier satisfies the constraints within the
  tolerance budget (this indicates a degenerate facet configuration).

For validation we compare this backend against the planar polygon formula and
the Lagrangian-product billiard solver on cubes, pentagon products, and random
low-facet seeds in the regression suite (`tests/math/test_capacity_ehz_haim_kislev.py`).

### Relation to Minkowski billiards

For Lagrangian products `Q × T ⊂ ℝ^4` with planar factors, the vertex‑contact
search above (≤3 bounces) computes the same capacity by the Artstein‑Avidan–
Ostrover program relating EHZ to closed Minkowski billiards. Rudolf’s theorem in
the plane guarantees the finite bounce bound, which is what the discrete search
exploits. This justifies the vertex‑billiard algorithm for the product case.

## Notes

- All functions are Torch‑first; return tensors and preserve dtype/device.
- Planar (2D) helpers are implemented; higher‑dimensional solvers are staged.
- A facet‑interior refinement of the billiard search is planned.
- `oriented_edge_spectrum_4d(vertices, normals, offsets)` — constructs the Chaidez–Hutchings oriented-edge graph of a 4D polytope, enumerates admissible cycles (respecting first-hit constraints on supporting facets), and returns the minimal combinatorial action. The solver backs `capacity_ehz_algorithm2` whenever the Lagrangian-product split is unavailable.
