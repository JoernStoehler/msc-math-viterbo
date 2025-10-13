---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Survey exact and approximate volume algorithms for convex polytopes in moderate dimension.
---

# Workflow: Algorithms for Volumes of Convex Polytopes

## Context

- Summarises strategies for computing or approximating $\operatorname{vol}(P)$ for convex polytopes in dimension $d = 2n \ge 4$.
- Each entry lists setting, main idea, references, and pseudo-code level procedure for reproducibility.

## Algorithms

### 1. Lawrence sign-decomposition (exact, general)

- **Reference.** Lawrence (1991), *Polytope volume computation*.
- **Idea.** Decompose $P$ into signed cones at each vertex and integrate their support functions.
- **Pseudocode.**
  ```text
  For each vertex v:
    Extract active constraints A_v x = b_v
    Form tangent cone C_v
    Accumulate signed volume contribution Δ_v
  Return Σ_v Δ_v
  ```

### 2. Beneath–beyond triangulation (exact, practical in low dimension)

- **References.** Avis & Fukuda (1992); Bueler, Enge & Fukuda (2000).
- **Idea.** Incrementally triangulate the polytope while enumerating facets/vertices; sum simplex volumes.

### 3. Barvinok short rational generating functions (exact for fixed $d$)

- **References.** Barvinok (1994); Barvinok & Pommersheim (1999).
- **Idea.** Express the indicator of $P$ as a short sum of rational generating functions and differentiate to recover volume.

### 4. Randomised hit-and-run approximation (FPRAS, general)

- **References.** Dyer, Frieze & Kannan (1991); Lovász & Vempala (2006).
- **Idea.** Use rapidly mixing random walks over a sequence of nested bodies to estimate volume ratios and obtain $(1+\varepsilon)$ approximations.
- **Pseudocode.**
  ```text
  Round P via affine transform
  Define cooling schedule of radii
  For each step:
    Sample N points with hit-and-run
    Estimate successive volume ratios
  Return baseline volume × product of ratios
  ```

### 5. Moment-based relaxations (approximate via convex optimisation)

- **Reference.** Lasserre (1998), *Volume of convex bodies and polynomial approximation*.
- **Idea.** Formulate volume as optimisation over moment sequences, solved via semidefinite programming; yields monotone bounds.

### 6. Specialised exact computation in 4D

- **Reference.** Gärtner, Santos & Ziegler (2003), *Exact volume computation for 4-polytopes with applications*.
- **Idea.** Exploit 4D facet–ridge graphs to build shellings and minimise intermediate complexity using exact arithmetic.

## Acceptance

- Keep algorithm summaries aligned with references cited in code comments and tests.
- Highlight dimensional regimes and complexity trade-offs so contributors can pick appropriate methods.

## Status Log

- 2025-02-14 — Migrated `docs/convex-polytope-volumes.md` into the briefs tree; maintained pseudo-code cues for implementation.
