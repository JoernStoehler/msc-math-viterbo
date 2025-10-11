---
status: draft
created: 2025-10-11
workflow: task
summary: Investigate symplectic-invariant similarity measures for convex polytopes and outline staged algorithmic deliverables.
---

# Context

- Focus: convex star-shaped polytopes in \(\mathbb{R}^{2n}\) (primary case \(2n = 4\)), compared up to affine symplectomorphisms **including translations**. Every algorithm must therefore recenter polytopes before applying linear symplectic transforms.
- Upstream references: see `docs/notes/symplectic-metrics.md` (capture derivations here if missing) and sampling guidance in `docs/algorithms/hit-and-run.md`.
- Desired outcomes per user stories: [S1] bulk pseudo-distances for UMAP, [S2] explicit symplectomorphism recovery for invariant verification, [S3] deduplication under symplectic equivalence, [S4] reduced representations usable in downstream ML, [S5] similarity search within a symplectic neighbourhood.
- Inputs: polytopes arrive in H- and/or V-representations; we assume access to volume estimators, Hit-and-Run sampling, Williamson canonicalisation, and matrix exponentials (for `exp(JK)` parameterisations).

# Objectives

1. Formalise translation-aware, symplectomorphism-invariant metrics and pseudo-metrics, together with robustness properties under facet perturbations and redundant inequalities.
2. Derive algorithmic blueprints for each user story, quantifying runtime, sample complexity, and error envelopes, and indicating which artefacts can be cached (≤1 kB per 4D polytope) versus recomputed on demand.
3. Specify the cached feature schema per algorithm (canonical transforms, spectral summaries, radial sketches, correlation histograms) without duplicating raw polytope data, and outline the persistence strategy.
4. Define evaluation criteria: correctness baselines (synthetic polytopes with known symplectic partners), numerical stability thresholds, throughput targets for batch scoring, and memory limits for caches.
5. Log open research questions needing maintainer sign-off (e.g., alternative canonical centres, acceleration libraries for optimisation).

# Analysis & mathematical notes

## 1. Canonicalisation pipeline (translations + Williamson)

- **Recentering**: use centroid translation \(t_A = \frac{1}{\operatorname{vol}(A)} \int_A x\, dx\) estimated via Hit-and-Run averages of vertices/samples. Subtracting \(t_A\) aligns translated copies.
- **Second moments**: compute \(M_A = \mathbb{E}[(x - t_A)(x - t_A)^\top]\); store as 4×4 SPD matrix (8 floating-point numbers in 4D once symmetry exploited).
- **Williamson normal form**: solve for symplectic \(S_A\) and diagonal \(D_A = \operatorname{diag}(d_{A,1},\dots,d_{A,n})\) s.t. \(S_A^{-\top} M_A S_A^{-1} = \operatorname{diag}(D_A, D_A)\). Implementation via eigendecomposition of \(J M_A\) followed by block QR.
- **Canonical coordinates**: map H-representation as \((A S_A^\top) y \le b\) to express \(\hat{A} = S_A^{-\top}(A - t_A)\). All downstream descriptors operate on \(\hat{A}\).
- **Caching**: store centroid (32 bytes), symmetric covariance (10 entries ≈ 80 bytes), symplectic transform (16 entries ≈ 128 bytes). Downstream descriptors live in dedicated caches: `RadialProfileCache` (≈1 kB), `CorrelationSketch` (≈0.5 kB), `SymplecticSpectrumCache` (≈200 B), `JaccardMonteCarloCache` (≈0.5 kB when seeded normals/offsets are amortised). Total footprint remains <1 kB per algorithm and ≈1.8 kB when all caches are present.

## 2. Symplectic Jaccard distance (Stories S2/S3 refinement)

- Definition: \(\delta_J(X,Y) = 1 - \frac{\operatorname{vol}(X \cap Y)}{\operatorname{vol}(X) + \operatorname{vol}(Y) - \operatorname{vol}(X \cap Y)}\). Symplectic variant \(d_J^{Sp}(A,B) = \inf_{S \in \operatorname{Sp}(2n,\mathbb{R})} \delta_J(A, S B)\) respects translations because \(\delta_J(A+t, SB+t) = \delta_J(A, SB)\).
- Optimisation: parameterise \(S = \exp(JK)\) for symmetric \(K\). Use stochastic approximation (Nesterov or Adam) on \(K\) with Monte Carlo intersection estimators using coupled random directions. For 4D: default `num_restarts=3`, `num_iterations=60`, `num_samples≈4e3`, learning rate 0.03.
- Estimation: evaluate probabilities \(p_A = \mathbb{P}[x \in SB]\) with CRNs; average with \(p_B\). Gradients use REINFORCE-style or smoothed indicator approximations. Deterministic seeding ensures cacheable score traces.
- Use cases: confirm invariants (S2), deduplicate (S3) after Stage-1 filtering, refine near neighbours (S5). Budget: ~150 ms/pair in 4D with JAX-on-GPU for final refinement.

## 3. Canonical radial \(L^p\) descriptor (Stories S1/S4/S5)

- Definition: \(r_A(u) = \sup\{ t \ge 0 : tu \in A \}\); compute softened min via `softmin_τ(b_j / max(⟨a_j,u⟩, ε))`. Work with \(\log r_A(u)\) for stability.
- Direction set: pre-sample \(M=128\) quasi-uniform directions on \(S^{2n-1}\) (shared across dataset). Evaluate on \(\hat{A}\) after canonicalisation to guarantee symplectic invariance.
- Distance: \(d_{r,p}(A,B) = ((1/M) \sum_k |\log r_A(u_k) - \log r_B(u_k)|^p)^{1/p}\) with \(p=2\) default. Optional minimisation over \(U(n) = Sp \cap O\) via small grid for extra alignment (cheap 2D rotations per \((q_i,p_i)\)-plane).
- Storage: 128 float64 values ≈ 1 kB; optionally compress using float32 for staging but keep float64 master copy for accuracy-critical workloads.
- Use cases: fast feature vectors for UMAP (S1) and ML (S4), coarse neighbour search (S5).

## 4. Symplectic correlation distance (Stories S1/S3/S5)

- For samples \(x_i, y_i \sim \operatorname{Unif}(A)\), compute \(z_i = x_i^\top J y_i\); the distribution is invariant under translations and symplectic transforms. Define \(d_{\mathrm{SCD}}(A,B) = W_1(\nu_A, \nu_B)\) where \(\nu_A\) is the law of \(z_i\).
- Estimators: (a) sorted sample mean absolute difference, (b) histogram EMD on fixed bins. Finite-sample error \(O(R^2 / \sqrt{N})\) with \(R\) radius bound; target `N=2000` pairs per polytope (≈16 kB raw, downsample to 64-bin histogram ≈ 512 bytes for cache).
- Provides deterministic lower bound on Jaccard: \(d_{\mathrm{SCD}} \le (2 R^2 / V) \operatorname{vol}(A \Delta B)\).

## 5. Symplectic spectrum feature (Stories S1/S4)

- Symplectic eigenvalues \(d_{A,i}\) from Williamson step yield spectrum vector \(d_A\). Feature distance `||log d_A - log d_B||_2`. In 4D this is two floats per polytope.
- Useful as stabiliser/invariant check, fast gating for far pairs. Combine with volume ratio to detect anisotropy.

## 6. Cascade strategy (Stories S1/S3/S5)

1. **Stage 0 (precomputation)**: assemble algorithm-specific caches—`SymplecticSpectrumCache`, `RadialProfileCache`, `CorrelationSketch`, optional `JaccardMonteCarloCache`—using shared canonicalisation artefacts. Total footprint ≈ 1.8 kB/polytope (fits user budget).
2. **Stage 1 (fast filter)**: compute `d_quick = 0.5 d_{r,2} + 0.35 d_{SCD} + 0.15 d_spec`. Thresholds: `τ_near = 0.08`, `τ_far = 0.6` (tune empirically). Flag near candidates for refinement and far candidates for rejection.
3. **Stage 2 (refinement)**: run symplectic Jaccard optimisation on near candidates. Optionally record optimal \(S\) to reuse in Story S2 checks.
4. **Stage 3 (alignment validation)**: if \(d_J^{Sp}\) below tolerance, recompute invariants on aligned copy to verify preserved symplectic quantities (capacity, EHZ, etc.).

## 7. Story-specific recommendations

- **S1 (UMAP embeddings)**: use cached log-radial vectors + spectrum as features; incorporate `d_{SCD}` when structure matters. Keep Stage-2 optional for interpretability drill-down.
- **S2 (explicit symplectomorphism)**: after Stage-2 Jaccard optimisation, store \(S\) and translation; validate by comparing symplectic capacities and other invariants on aligned bodies.
- **S3 (deduplication)**: Stage-1 rejects; Stage-2 accept if `d_J^{Sp} < 1e-3` and `d_{SCD} < 1e-4`. Maintain canonical hash (rounded transform + log radial) for quick lookups.
- **S4 (reduced representations)**: canonical coordinates + log radial vector provide symplectic invariants; optionally append spectrum and volume for ML input.
- **S5 (near neighbour search)**: index log-radial vectors with approximate nearest neighbour (e.g., FAISS). For borderline pairs run Stage-2 to confirm closeness.

# Execution

1. **Literature and internal review**
   1.1 Formalise derivations above into `docs/notes/symplectic-metrics.md`, citing symplectic geometry texts (de Gosson, Folland) and measure transport references for Wasserstein bounds.
   1.2 Validate translation handling analytically (centroid vs. symplectic barycentre) and numerically on synthetic translations.
   1.3 Document parameter defaults and convergence heuristics for each metric.
2. **Algorithm design per story**
   2.1 Create comparative tables (runtime/sample complexity, error bounds, cache footprint) for S1–S5.
   2.2 Specify deterministic RNG seeding plan (per-polytope base seed, per-pair counter-based splits) enabling cache reuse.
   2.3 Determine fallback heuristics when canonicalisation fails (e.g., nearly singular moments) and escalate for maintainer guidance.
3. **Data artefact specification**
   3.1 Define serialization hooks (msgpack or npz) for each cache dataclass (`SymplecticSpectrumCache`, `RadialProfileCache`, `CorrelationSketch`, `JaccardMonteCarloCache`), respecting the ≤1 kB budget per algorithm.
   3.2 Define invalidation policy when polytope geometry changes (recompute shared canonicalisation artefacts, lazily refresh algorithm caches).
4. **Validation plan**
   4.1 Build synthetic suite: axis-aligned boxes, ellipsoids, random polytopes, symplectically rotated copies, and perturbations with redundant facets.
   4.2 Establish accuracy thresholds (e.g., `d_{r,2}` stable within 1e-3 under 1e-2 facet noise) and runtime goals (<30 µs for Stage-1 distance on cached data).
   4.3 Prototype UMAP embedding experiment to confirm clustering of symplectic orbits; measure dedup recall/precision on constructed dataset.
5. **Deliverables & sequencing**
   5.1 Phase A: implement canonicalisation utilities + caching; Phase B: implement descriptors & pairwise distances with unit tests; Phase C: integrate cascade into dataset workflows; Phase D: document usage and benchmarking results.
   5.2 Record blockers (e.g., need for manifold optimisation tooling) and escalate if dependencies exceed standard stack.

# Dependencies / Unlocks

- **Dependencies**: Hit-and-Run sampler (with translation support), robust volume estimation, linear algebra for Williamson, Wasserstein solvers (1D EMD), ANN backend for Stage-1 features.
- **Unlocks**: symplectic-invariant ML features, dedup tooling, interpretability dashboards, validation harness for symplectic invariants.

# Status log

- 2025-10-11 — Expanded mathematical analysis drafted; API stubs and failing tests added for upcoming implementations.
- 2025-10-12 — Updated plan for translation-aware canonicalisation in the modern namespace with per-algorithm caches and paired direct/cached evaluators.

# Open questions

1. Choose the canonical centre: compare centroid, symplectic barycentre, and John ellipsoid centre for numerical stability; document trade-offs before coding.
2. Calibrate Stage-1 thresholds per dataset size (UMAP vs dedup) and decide whether to adaptively tune weights/thresholds online.
3. Evaluate whether histogram compression (e.g., quantile sketch) suffices for correlation distribution while staying within the 1 kB budget.
