# Algorithms for $c\_{\\mathrm{EHZ}}$ of Convex Polytopes

We collect high-level procedures for evaluating the Ekeland–Hofer–Zehnder (EHZ) capacity of convex
polytopes in $\\mathbb{R}^{2n}$ with $2n\\ge 4$. Each algorithm is described at the pseudo-code
level with references for proofs, correctness, and complexity.

## 1. Facet-Normal Optimization Formula (General Dimension)

**Reference.** K. Leipold and F. Vallentin, "Computing the EHZ capacity is NP-hard," _Journal of
Symplectic Geometry_ (2024), Theorem 1.1.【F:docs/convex-polytope-cehz-capacities.md†L8-L35】

**Idea.** Express $c\_{\\mathrm{EHZ}}(P)$ as an optimization problem over facet normals $n_i$ with
coefficients $\\beta_i \\ge 0$ satisfying $\\sum \\beta_i n_i = 0$; minimize a quadratic form
derived from support numbers. Although NP-hard in general, the formulation enables exact solving for
small instances and serves as the basis for relaxations.

```text
Input: Polytope P = {x | A x ≤ 1} with facet normals n_i
Output: Optimal value c_EHZ(P)
1. Collect facet normals {n_1, …, n_k} and support values h_P(n_i) = 1.
2. Solve the optimization problem:
       minimize    ½ Σ_{1 ≤ j < i ≤ k} β_i β_j ω(n_i, n_j)
       subject to  β ∈ ℝ^k_{≥0},  Σ_{i=1}^k β_i n_i = 0,
   where ω is the standard symplectic form on ℝ^{2n}.
3. Return c_EHZ(P) = optimal objective value.
```

## 2. Mixed-Integer Linear Programming Relaxation (General Dimension)

**Reference.** A. Krupp, _Calculating the EHZ Capacity of Polytopes_, Ph.D. thesis, Universität zu
Köln (2020), Chapter 4.【F:docs/convex-polytope-cehz-capacities.md†L37-L64】

**Idea.** Linearize the facet-normal optimization by introducing binary variables encoding which
facets support the closed characteristic, leading to a MILP that bounds $c\_{\\mathrm{EHZ}}(P)$ and
is exact when the active facet set matches the true minimizer.

```text
Input: Polytope P = {x | A x ≤ b}
Output: Upper and lower bounds (optionally exact) for c_EHZ(P)
1. Enumerate candidate facet sets F ⊆ {1,…,k} up to size 2n.
2. For each F, set up MILP variables:
       β_i ≥ 0 for i ∈ F, binary y_i indicating activation.
3. Impose constraints:
       Σ_{i∈F} β_i n_i = 0,   β_i ≤ M y_i,
       Σ_{i∈F} y_i = 2n,      support consistency A_F^T λ = 0.
4. Minimize ½ Σ_{i<j} β_i β_j ω(n_i, n_j) subject to MILP constraints.
5. Aggregate best feasible objective over explored facet sets.
```

## 3. Combinatorial Reeb Orbit Enumeration (4 Dimensions)

**Reference.** D. Chaidez and M. Hutchings, "Computing Reeb dynamics on four-dimensional convex
polytopes," _Advances in Mathematics_ 404 (2022), Theorem
1.2.【F:docs/convex-polytope-cehz-capacities.md†L66-L96】

**Idea.** In dimension four ($n=2$), closed Reeb orbits correspond to combinatorial cycles on the
$1$-skeleton. Enumerate admissible cycles, compute their combinatorial action, and take the minimum.

```text
Input: 4-dimensional convex polytope P with rational vertices
Output: Exact c_EHZ(P)
1. Construct the directed graph whose vertices are oriented edges of P.
2. Build transition rules encoding Reeb admissibility across adjacent facets.
3. Enumerate primitive cycles γ satisfying admissibility.
4. For each γ, compute action A_comb(γ) from facet normals and support numbers.
5. Return min_γ A_comb(γ).
```

_This algorithm is specific to 4D; higher dimensions require additional combinatorics not covered by
Chaidez–Hutchings._

## 4. Minkowski Billiard Shortest Path Search (Lagrangian Products)

**Reference.** D. Rudolf, "The Minkowski billiard characterization of the EHZ-capacity of convex
Lagrangian products," _Journal of Dynamics and Differential Equations_ 34 (2022), Theorem
1.1.【F:docs/convex-polytope-cehz-capacities.md†L98-L126】

**Idea.** For $P = K \\times T$ with convex polytopes $K, T \\subset \\mathbb{R}^n$,
$c\_{\\mathrm{EHZ}}(P)$ equals the length of the shortest closed $(K,T)$-Minkowski billiard
trajectory. This reduces computation to a discrete geodesic problem on the normal fan of $K$ (or
$T$).

```text
Input: Polytopes K, T ⊂ ℝ^n
Output: c_EHZ(K × T)
1. Construct facet normal fans of K and T.
2. Enumerate combinatorial billiard paths of length m ≤ n+1 through the normal fan.
3. For each path, compute Minkowski length ℓ_T(γ) using support functions of T.
4. Return minimal ℓ_T(γ) over admissible closed paths.
```

_Calibration._ For centrally symmetric factors the extremal trajectories appear at four reflections.
Rudolf works out the square–diamond Hanner pair, exhibiting a four-bounce $(K,T)$-orbit of
$\ell_T$-length $8$ that realises $c_{\mathrm{EHZ}}([-1,1]^2 \times B_1^2)$ and matches the
Artstein-Avidan–Ostrover bounds.【F:docs/convex-polytope-cehz-capacities.md†L111-L121】We use this
configuration as the canonical baseline when validating the reference and fast solvers and enforce a
minimum of three distinct reflections in the enumeration to exclude degenerate two-bounce
paths.【F:tests/viterbo/symplectic/capacity/minkowski_billiards/test_minkowski_billiards.py†L27-L65】
【F:src/viterbo/symplectic/capacity/minkowski_billiards/reference.py†L20-L46】

## 5. Convex Programming Relaxations via Support Functions (General Dimension)

**Reference.** B. Haim-Kislev, "Symplectic capacities of convex polytopes via support functions,"
_Selecta Mathematica_ 29 (2023), Proposition
3.4.【F:docs/convex-polytope-cehz-capacities.md†L128-L155】

**Idea.** Approximate $c\_{\\mathrm{EHZ}}(P)$ by smoothing the support function of $P$, solving the
resulting convex optimization problem for closed characteristics, and refining via limit arguments.
Provides convergent upper bounds and guides numerical continuation.

```text
Input: Polytope P, smoothing parameter ε > 0
Output: Upper bound U_ε ≥ c_EHZ(P)
1. Smooth support function h_P via convolution to obtain h_{P,ε} with C^1 boundary.
2. Solve convex optimization for minimal action closed characteristic on ∂P_ε.
3. Record action value U_ε.
4. Decrease ε and repeat; the sequence U_ε ↓ c_EHZ(P).
```

## 6. Symmetry-Reduced Search for Centrally Symmetric Polytopes

**Reference.** S. Artstein-Avidan and Y. Ostrover, "Symplectic billiards and symplectic capacities,"
_Duke Mathematical Journal_ 139 (2007), §4.【F:docs/convex-polytope-cehz-capacities.md†L157-L185】

**Idea.** For centrally symmetric polytopes, $c\_{\\mathrm{EHZ}}$ equals the action of symmetric
billiard trajectories. Reduce the facet-normal optimization by grouping opposite facets, yielding
smaller linear systems.

```text
Input: Centrally symmetric polytope P with opposite facet pairs (F_i, F_i')
Output: c_EHZ(P)
1. For each facet pair, enforce β_i = β_i'.
2. Solve reduced optimization:
       minimize ½ Σ_{i<j} β_i β_j ω(n_i, n_j)
       subject to Σ_i β_i (n_i - n_i') = 0, β_i ≥ 0.
3. Evaluate candidate symmetric billiard orbits to confirm minimality.
```

_Symmetry reduction does not change the optimal value but significantly reduces search space in
practice._
