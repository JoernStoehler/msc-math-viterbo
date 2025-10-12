---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Catalogue algorithms and references for computing the EHZ capacity of convex polytopes.
---

# Workflow: Algorithms for $c_{\mathrm{EHZ}}$ of Convex Polytopes

## Context

- Covers high-level procedures for evaluating the Ekeland–Hofer–Zehnder (EHZ) capacity $c_{\mathrm{EHZ}}$ on convex polytopes in $\mathbb{R}^{2n}$.
- Implementations live in `viterbo.capacity`; tolerance constants are centralised in `src/viterbo/numerics.py` to maintain consistency across capacity, cycle, and spectrum stacks.
- The legacy `viterbo.symplectic` namespace has been retired; use the modern modules directly.

## Algorithms

### 1. Facet-normal optimisation (general dimension)

- **Reference.** Leipold & Vallentin (2024), *Computing the EHZ capacity is NP-hard*, Theorem 1.1.
- **Idea.** Optimise over non-negative coefficients $\beta_i$ on facet normals with $\sum \beta_i n_i = 0$ and minimise a quadratic form derived from support numbers.
- **Usage.** Exact for small instances; anchors relaxations for larger polytopes.
- **Pseudocode.**
  ```text
  Input: P = {x | A x ≤ 1}
  Collect normals {n_i}
  Solve: minimize ½ Σ_{i<j} β_i β_j ω(n_i, n_j)
          subject to β ≥ 0, Σ β_i n_i = 0
  Return optimal value
  ```

### 2. Mixed-integer linear relaxation (general dimension)

- **Reference.** Krupp (2020), PhD thesis *Calculating the EHZ Capacity of Polytopes*, Chapter 4.
- **Idea.** Linearise the facet-normal programme by activating candidate facet sets with binary variables; yields bounds and sometimes exact values.
- **Pseudocode.**
  ```text
  Enumerate facet sets F up to size 2n
  Introduce β_i ≥ 0 and binaries y_i for i ∈ F
  Constraints: Σ β_i n_i = 0, β_i ≤ M y_i, Σ y_i = 2n
  Minimise ½ Σ_{i<j} β_i β_j ω(n_i, n_j)
  ```

### 3. Combinatorial Reeb orbit enumeration (4D)

- **Reference.** Chaidez & Hutchings (2022), *Computing Reeb dynamics on four-dimensional convex polytopes*, Theorem 1.2.
- **Idea.** Build the oriented-edge graph of $P$, enumerate admissible cycles representing Reeb orbits, and take the minimal combinatorial action.
- **Notes.** Specific to $n=2$; integrates with the shared graph builder in `viterbo.capacity.reeb_cycles`.

### 4. Minkowski billiard search (Lagrangian products)

- **Reference.** Rudolf (2022), *The Minkowski billiard characterization of the EHZ-capacity of convex Lagrangian products*, Theorem 1.1.
- **Idea.** For $P = K \times T$, $c_{\mathrm{EHZ}}(P)$ equals the length of the shortest closed $(K,T)$-Minkowski billiard trajectory.
- **Calibration.** Square–diamond Hanner pair gives a canonical baseline with length $8$; enforced in tests to avoid degenerate two-bounce paths.

### 5. Support-function smoothing and convex relaxations (general dimension)

- **Reference.** Haim-Kislev (2023), *Symplectic capacities of convex polytopes via support functions*, Proposition 3.4.
- **Idea.** Smooth the support function, solve for closed characteristics on the smoothed boundary, and decrease the smoothing parameter to converge from above.

### 6. Symmetry-reduced search (centrally symmetric polytopes)

- **Reference.** Artstein-Avidan & Ostrover (2007), *Symplectic billiards and symplectic capacities*, §4.
- **Idea.** Pair opposite facets to halve the variable count and restrict to symmetric billiard trajectories, reducing the search space without changing the optimum.

## Acceptance

- Brief remains synchronised with implemented solvers and corresponding tests (e.g., `tests/viterbo/test_capacity_solvers.py`).
- Each algorithm lists a reproducible reference and highlights constraints or dimensional specialisations.

## Status Log

- 2025-02-14 — Migrated `docs/convex-polytope-cehz-capacities.md` into the briefs tree; emphasised calibration tests and namespace updates.
