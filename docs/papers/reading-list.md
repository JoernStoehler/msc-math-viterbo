# Reading List — Symplectic Embeddings, Capacities, and ML

This list collects papers, preprints, and code that are directly or tangentially relevant to symplectic capacities (esp. EHZ), systolic ratio on convex polytopes, and data/ML tooling. Each entry notes what kind of content it provides so readers can jump in appropriately.

## Core Algorithms and Computation (EHZ / Minimal Action)

- Pazit Haim‑Kislev — “EHZ capacity of convex polytopes” (paper)
  - Link: https://link.springer.com/article/10.1007/s00039-019-00486-4
  - Type: algorithm/theory; minimal action on convex polytopes in R^{2n}.
  - High‑level: Defines and analyzes an algorithm to compute the EHZ capacity (minimal action) for convex polytopes. Practical complexity grows roughly like F! (F = number of facets). Useful for verified small‑F computations and as a baseline for checks.

- Chaidez & Hutchings — “EHZ capacity of polytopes in R^4 via action and CZ index” (paper)
  - Link: https://www.aimsciences.org/article/doi/10.3934/jcd.2021016
  - Type: algorithm/theory; 4D specialization.
  - High‑level: Alternative computational framework for EHZ in 4D using action and Conley–Zehnder index. Includes examples; reports polytopes with systolic ratio 1 (e.g., the 24‑cell). Good reference for 4D‑specific techniques and invariants.

- Rudolf & Krupp — “EHZ capacity for Lagrangian products of convex polytopes” (paper)
  - Link: https://link.springer.com/article/10.1007/s10711-022-00717-3
  - Type: algorithm/theory + code reference.
  - High‑level: Computes EHZ capacity for Lagrangian products using a characterization of shortest billiard trajectories by Bezdek & Bezdek. Good for product‑structured instances and for mapping to Minkowski billiard dynamics.

## Complexity and Limitations

- Leipold & Vallentin — “Computing the EHZ capacity of simplices is NP‑hard” (preprint)
  - Link: https://arxiv.org/abs/2402.09914
  - Type: complexity.
  - High‑level: Shows NP‑hardness for simplices in 2n dimensions. Motivates specialized low‑dimensional (e.g., 4D) methods, surrogates, and careful dataset design.

## Code Repositories (Baseline Implementations)

- Haim‑Kislev (MATLAB) — EHZ capacity of convex polytopes
  - Link: https://github.com/pazithaimkislev/EHZ-capacity
  - Type: reference implementation.
  - High‑level: MATLAB code computing EHZ capacity/minimal action on polytopes. Fast for small facet counts; a baseline to cross‑check small instances.

- Rudolf–Krupp (Python) — EHZ capacity for polytopes (products)
  - Link: https://github.com/S-Krupp/EHZ-capacity-of-polytopes
  - Type: reference implementation (product cases).
  - High‑level: Code accompanying their approach to Lagrangian product polytopes; useful for product‑structured families and validation.

## Symplectic Embeddings (General Context)

- Schlenk — “On symplectic folding” (1999)
  - Link: https://arxiv.org/abs/math/9903086
  - Type: theory/surveys of embeddings techniques.
  - High‑level: Foundational techniques and constructions for embeddings of simple shapes. Good background on rigidity vs. flexibility and near‑optimal constructions.

- Hind — “Some optimal embeddings of symplectic ellipsoids” (2014)
  - Link: https://arxiv.org/abs/1409.5110
  - Type: theory; explicit embeddings.
  - High‑level: Constructions for higher‑dimensional ellipsoids into product targets (ball/cube × Euclidean space); informative for quantitative embedding thresholds.

- Hutchings et al. — “Anchored symplectic embeddings” (2024)
  - Link: https://arxiv.org/abs/2407.08512
  - Type: theory; 4D toric domains with anchoring constraints.
  - High‑level: Introduces anchored embeddings; shows quantitative criteria and distinctions from ordinary embeddings. Contextualizes constraints that alter embedding feasibility.

## Systolic Ratio and Related Geometry

- Ostrover & collaborators — works on systolic ratio for convex polytopes (various)
  - Type: theory/examples; counterexamples context.
  - High‑level: Results and examples around systolic ratio in convex settings, including the canonical 4D constructions. Use as conceptual background for goals (sys ≥ 1 and beyond).

- Reeb orbits on surfaces (various sources)
  - Type: background on minimal action cycles and genericity.
  - High‑level: Guidance on where minimal action cycles may lie in generic settings; informs search strategies and validation tests.

## ML‑Adjacent / Optimization on Symplectic Structures

- Gajewski, Goldin, Safin, Singh, Zhang — “Optimization on Symplectic Embeddings” (private preprint, July 2019)
  - Status: private preprint (no arXiv record under this title at the time of writing).
  - High‑level: Mentions neural networks/learning in the context of symplectic embeddings; potential ML‑based optimization of embeddings. Treat as non‑public unless authors consent.

- Symplectic structure‑aware learning (example): Liu, Yi, Zhang, Huang — “Symplectic Structure‑Aware Hamiltonian (Graph) Embeddings” (2023–2024)
  - Link: https://arxiv.org/abs/2309.04885
  - Type: ML; structure‑aware optimization on symplectic manifolds (graph setting).
  - High‑level: Illustrates how symplectic constraints/manifolds can be integrated into optimization/learning; useful inspiration for geometry‑aware model design.

## What to Consult Each For

- Need a verified EHZ/minimal‑action computation for small polytopes? → Haim‑Kislev paper + MATLAB code.
- Working in 4D and want an index/action‑based method? → Chaidez–Hutchings.
- Lagrangian products or billiard‑linked cases? → Rudolf–Krupp (+ Bezdek & Bezdek characterization).
- Wondering about limits of exact computation? → Leipold–Vallentin (NP‑hardness for simplices).
- Seeking embedding constructions and constraints context? → Schlenk; Hind; Hutchings et al. (anchored embeddings).
- Framing ML approaches or structure‑aware models? → Gajewski et al. (private preprint); Liu et al. (graph embeddings on symplectic manifolds).

## Notes and To‑Do

- Verify and add precise citations for: (a) Ostrover/Kislev works on systolic ratio counterexamples; (b) Reeb‑orbits background best references; (c) Mahler conjecture status in 4D/6D.
- When internal implementations mature, add cross‑refs to `src/viterbo/math/` modules and comparison plots.

