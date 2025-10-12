---
title: "Experiments: Symplectic-Invariant Encodings for Convex Polytopes"
status: proposed
version: 1.0
last_updated: 2025-10-12
owners: ["Assigned Developer"]
reviewers: ["Research Lead", "ML Lead"]
stage: implementation-brief
depends_on: ["docs/briefs/polytope-encodings.md"]
deliverables:
  - "Reusable encoding library (Python) with CLI + unit tests"
  - "Training scripts for baselines; logged metrics + artifacts"
  - "Evaluation report (markdown) with tables/plots"
tags: [implementation, evaluation, python, pytorch, geometry, symplectic, OT, MMD]
---

# 0. TL;DR

Implement and evaluate three baseline encoders for convex polytopes under affine symplectomorphisms. Each baseline must support (i) canonicalization, (ii) robust sampling/encoding, (iii) invariance/teacher losses, and (iv) standard metrics. Produce a one-click evaluation that outputs tables/plots and a markdown report we can commit to the repo.

---

# 1. Repository layout

```

polytope_encodings/
canonicalize/
williamson.py             # symplectic eigenvalues & canonical frame
center.py                 # centroid, Chebyshev center fallback
geometry/
reps.py                   # V-rep/H-rep utilities; convex hull ops
support.py                # h_P(u) via vertices or LP fallback
measure.py                # facet normals/areas, surface measure
sampling.py               # HEALPix & Fibonacci sphere grids
encoders/
support_sphere/
spherical_cnn.py        # optional; else SH spectra + MLP
spectra.py              # SH power/bispectra utils
surface_measure/
smooth_field.py         # kernel smoothing on S^{d-1}
kme_mlp.py              # kernel mean embedding + MLP
set_transformer/
set_encoder.py          # over {(u,h_P(u))} or {(n_i,b_i)}
training/
losses.py                 # InfoNCE, triplet, regression
teachers.py               # support L^p, Sinkhorn OT, sliced OT, MMD
augment.py                # S=exp(H), translations, noise/duplications
eval/
invariance.py             # group invariance tests
continuity.py             # perturbation stress tests
retrieval.py              # NN retrieval vs teacher
tasks.py                  # volume/facet-count regression
report.py                 # plots + markdown tables
scripts/
train_baseline_*.py
eval_all.py
tests/
test_williamson.py
test_support_fn.py
test_ot_mmd.py
...

````

---

# 2. Canonicalization (must-have)

1. **Centering:** translate \(P\) so the centroid is at the origin. If centroid is ill-conditioned (thin polytopes), fall back to Chebyshev center of \(P\).
2. **Williamson frame:**  
   - Compute the (approximate) second-moment matrix \(M=\frac{1}{|K|}\int_K xx^\top dx\). For polytopes, approximate by sampling or by triangulation (volume weights).  
   - Compute **symplectic eigenvalues** and a **symplectic** \(S\) with \(S^\top M S = D\oplus D\). Map \(P\mapsto \tilde P=S^{-1}P\).  
   - Export \(\{d_i\}\) as additional features (they are symplectic invariants).  
   References: [Williamson1936], [Bhatia2018].

3. **Optional learned canonicalizer:** a small network predicts \(H\in\mathfrak{sp}(2n)\) (Hamiltonian) from a coarse encoding; apply \(S=\exp(H)\). Regularize with \(\|S^\top J S-J\|\). Use only in the “+LC” variants. The fact \(\exp(H)\in \mathrm{Sp}(2n)\) is standard [HamiltonianMatrixWiki].

---

# 3. Sphere sampling

- Implement **Fibonacci** and **HEALPix** grids for \(S^{d-1}\) with APIs:
  ```python
  U = fibonacci_sphere(n_points, d=4)  # returns (N, d) unit vectors
  U = healpix_grid(nside, d=4)         # if using HEALPix-like in 4D, fallback to geodesic icosahedral grids for d=3 and cartesian products for d>3
````

* Verify near-uniformity by empirical area tests [Gonzalez2009], [Gorski2005], [SaffKuijlaars1997].

---

# 4. Teachers (for contrastive/metric learning)

Implement three interchangeable teacher distances:

* **Support (L^2)/(L^\infty)** on (S^{d-1}): sample (h_P(u)), (h_Q(u)) and compute ( |h_P-h_Q|_p); optionally optimize over a small set of learnable symplectic transforms (S) (or a few Gauss–Newton steps in (\mathfrak{sp}(2n))).
* **Sinkhorn OT** (grid-based) between smoothed surface measures (S_P,S_Q) on the sphere with geodesic-cost matrix; use POT or a light custom implementation [Cuturi2013], [PeyreCuturi2019].
* **Sliced OT on (S^2)** (if (d=3)); otherwise standard sliced OT in (\mathbb{R}^{2n}) as a fallback [Quellmalz2023,2024].

Also implement **MMD** with vMF/heat kernels on the sphere [Sriperumbudur2010], [Gretton2012].

---

# 5. Baselines to build

## Baseline 1 — **Support↦Sphere**

* Pipeline: canonicalize (\rightarrow) sample (h_P) on (U) (\rightarrow) encoder:

  * **1a (no external deps):** compute spherical-harmonic **power spectra** up to band (L) (approximate via least-squares on (U)); feed to an MLP.
  * **1b (if allowed):** Spherical CNN on the sampled field defined on (U) [Cohen2018], [Esteves2018].
* Losses: InfoNCE/triplet with **support** and **OT/MMD** teachers + **invariance penalty** under random (S=\exp(H)), translations.

## Baseline 2 — **SurfaceMeasure↦Sphere**

* Compute facet normals/areas; form discrete measure (S_P=\sum a_i\delta_{n_i}); smooth via spherical heat/vMF kernel to a field; encode with Spherical CNN **or** kernel mean embedding + MLP [Muandet2017].
* Train as in Baseline 1.

## Baseline 3 — **Set Transformer**

* Inputs: either ({(u,h_P(u))}) or ({(n_i,b_i)}).
* Model: Set Transformer with inducing points; duplicate/near-parallel suppression via attention; pooling gives permutation invariance [Lee2019].
* Train as above.

**Fusion (optional):** concatenate Baseline 1 & Baseline 3 embeddings; small fusion MLP.

---

# 6. Datasets & generation

* **Synthetic generator:** random H-polytopes (Gaussian facet normals with random offsets producing bounded (P)); V-polytopes from random points followed by convex hull; apply random ASp transforms; add controlled noise (near-duplicate facets/vertices, small translations/rotations).
* **Splits:** train/val/test with non-overlapping seeds and transform ranges.

---

# 7. Metrics & acceptance criteria

For each baseline (and fusion), report:

1. **ASp-invariance score.** For (K) random symplectic transforms and (T) translations per test polytope, compute
   ( \mathrm{Inv} = \frac{1}{KT}\sum |E(P)-E(SP+t)|_2 ).
   **Target:** monotone decrease with training; < 0.15 (normalized) on test.

2. **Continuity score.** For perturbations (\Delta), compare code drift vs. teacher distance: Spearman (\rho(E\text{-dist}, d_\text{teacher})) > 0.9.

3. **Retrieval (NN).** Recall@10 ≥ 0.9 when ground-truth neighbors are defined by teacher distances.

4. **Regression sanity.** (R^2!\ge!0.95) for (\log \mathrm{vol}) on held-out; stable for facet-count histograms.

5. **Efficiency.** Encode 1k 4D polytopes with ≤5k facets each in ≤ X minutes on 1 GPU / CPU-only fallback within 3×.

6. **Ablations.** Demonstrate benefit of Williamson canonicalization (> +5 points Recall@10 over no-canon).

A run passes if **≥4/6** targets are met for at least one baseline.

---

# 8. Implementation notes

* **Support function.** If V-rep exists, (h_P(u)=\max_j \langle v_j,u\rangle). If only H-rep ( {x:, n_i^\top x\le b_i}), evaluate by LP (small) or compute vertices once (Qhull/pyhull) then reuse.
* **Symplectic eigenvalues & (S).** Use eigendecomposition of (iJM) or a dedicated routine (see [Bhatia2018]); add unit tests with known ellipsoids.
* **OT.** Start with small grids (e.g. (N\le 2048) directions) for speed; use (\epsilon)-regularization and log-domain stabilization [Cuturi2013].
* **Augmentations.** Sample (H) by drawing a random symmetric (K) and setting (H=JK) so (H\in\mathfrak{sp}(2n)); set (S=\exp(H)) (use `scipy.linalg.expm`).
* **Stability.** Apply small Gaussian jitter to vertices/offsets; randomly add/remove duplicates during training.

---

# 9. Experiments to run

1. **Grid resolution study:** (N\in{512, 2{,}048, 8{,}192}) (Fibonacci), (L\in{6,10,14}) for SH.
2. **Teacher choice:** support-(L^2) vs. Sinkhorn vs. sliced OT; compare wall-time and retrieval quality.
3. **Canonicalization ablation:** no canon / Williamson only / + learned canonicalizer.
4. **Representation ablation:** support vs surface-measure vs set-only; and fusion.
5. **Noise robustness:** duplicate facets, near-coplanar merges, vertex jitter.
6. **(Optional) 4D capacity proxy:** where feasible, compute small-batch EHZ or known proxies to correlate with embeddings (note NP-hardness in general [LeipoldVallentin2024AMS]).

---

# 10. Reporting

Run `scripts/eval_all.py` to produce:

* `reports/metrics.csv` with invariance/continuity/retrieval/efficiency
* `reports/plots/*.png` (invariance vs. epoch; Recall@k; speed vs. grid size)
* `docs/briefs/polytope-encodings-eval.md` (autogenerated; tables + brief narrative)

---

# 11. External references (for developer convenience)

* **Convex geometry & support functions:** [Schneider2014], [SupportFnWiki], [Kousholt2015]
* **Spherical encoders:** [Cohen2018], [Esteves2018], [Kazhdan2003]
* **Set encoders:** [Zaheer2017], [Lee2019], [Qi2017]
* **OT:** [Cuturi2013], [PeyreCuturi2019], [Quellmalz2023]
* **Canonicalization:** [Williamson1936], [Bhatia2018], [HamiltonianMatrixWiki]

(See full citations in the research brief.)

```
---

If you’d like, I can also generate a small “seed” dataset script and a starter `requirements.txt` (with toggles for optional Spherical CNNs vs. SH-only) so a dev can run Baseline 1 end-to-end immediately.
```
