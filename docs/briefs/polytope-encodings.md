---
title: "Representing Convex Polytopes for ML under Affine Symplectomorphisms"
status: ready
version: 1.0
last_updated: 2025-10-12
owners: ["ML+Geometry"]
reviewers: ["Research Lead", "ML Lead"]
stage: research-brief
tags: [convex-geometry, symplectic, polytopes, support-function, surface-measure, set-encoders, spherical-CNN, transformers, OT, MMD]
summary: >
  This brief consolidates design options and evaluation criteria for
  ML-ready encodings of convex polytopes in R^{2n} that are
  (as far as practical) invariant to affine symplectomorphisms,
  permutation/augmentation of V- or H-representations, and continuous
  under small geometric perturbations.
---

# 1. Problem & constraints

We need encodings for convex polytopes \(P\subset\mathbb{R}^{2n}\) (typically \(n=2\) or \(3\)) that:

- **Identify shapes up to affine symplectomorphisms** \(x\mapsto Sx+t\) with \(S\in \mathrm{Sp}(2n,\mathbb{R})\) and \(t\in\mathbb{R}^{2n}\).
- Are **permutation/augmentation-invariant** to ordering of vertices/facets, duplicate/near-duplicate elements, extra interior points or redundant halfspaces.
- Are **continuous/stable** w.r.t. small perturbations of vertices/halfspaces, or small changes such as adding points just outside a facet.
- Are amenable to **GNN/Transformer**-style models, and support downstream tasks (retrieval, metric learning, regression of invariants, decoding to a representative polytope).

Exact, cheap invariance to the full affine symplectic group is unlikely; we pursue **canonicalization + invariant/robust encoders + augmentation + teacher distances**.

---

# 2. Encoders that buy the right invariances

## 2.1 Signals/measures on the sphere (recommended)

**(A) Support function \(h_P\).** Sample \(h_P(u)=\max_{x\in P}\langle x,u\rangle\) on a near-uniform grid \(U\subset S^{2n-1}\) (HEALPix, Fibonacci). The support function **uniquely determines** a convex body and **metrizes** Hausdorff distance via
\(d_H(P,Q)=\|h_P-h_Q\|_{\infty}\); it is naturally **permutation/augmentation-invariant** and **continuous** in \(P\) [Schneider2014][SchneiderCambridge], [SupportFnWiki], [Kousholt2015].  
Model options: Spherical CNNs or spherical-harmonic spectra (rotation-invariant power spectrum), or a Set/Transformer over \(\{(u, h_P(u))\}\) with positional embeddings on the sphere [Cohen2018], [Esteves2018], [Kazhdan2003].

**(B) Surface area measure \(S_P\).** A discrete measure on \(S^{2n-1}\) supported at facet normals with weights=facet areas. By Minkowski’s existence theorem, \(S_P\) characterizes \(P\) up to translation; it’s a **finite measure** representation well-suited to kernel mean embeddings and optimal transport (OT) distances on the sphere [Schneider2014].  
Model options: smooth \(S_P\) by a heat/vMF kernel to a field on the sphere and apply a Spherical CNN; or embed \(S_P\) via **kernel mean embeddings** (MMD) [Sriperumbudur2010], [Muandet2017], [Gretton2012].

**Sampling grids.** Use HEALPix (equal-area, fast SH) or Fibonacci spheres; both give uniform-ish coverage and stable spectra [Gorski2005], [Gonzalez2009], [SaffKuijlaars1997].

## 2.2 Set/graph encoders (auxiliary or alternative)

- **Set encoders** (Deep Sets, Set Transformer) over facets \((n_i,b_i)\) or vertices \(v_j\); permutation-invariant and robust when trained with duplication/noise augmentations [Zaheer2017], [Lee2019], [Qi2017].
- **Incidence/bipartite graphs** (facet–vertex) using GNN/graph Transformers. Useful for combinatorics, but beware **combinatorial instability** under tiny perturbations—keep as an auxiliary channel [GraphTransformerSurvey2024].

## 2.3 Implicit neural representations

Occupancy/SDF fields (DeepSDF, Occupancy Networks, SIREN) are continuous by construction; for convex polytopes, couple with linear programs to query \(h_P\) in selected directions or to constrain convexity via differentiable optimization layers [Park2019], [Mescheder2019], [Sitzmann2020], [Agrawal2019].

---

# 3. Getting (approximate) affine-symplectic invariance

We use **canonicalize \(\rightarrow\) encode \(\rightarrow\) group augmentation**:

1. **Translation**: subtract a stable center (e.g., centroid/Chebyshev center).
2. **Symplectic canonicalization (Williamson).** Compute the second-moment/inertia matrix \(M=\mathbb{E}[xx^\top]\) (boundary- or volume-weighted). By **Williamson’s theorem**, \(\exists S\in\mathrm{Sp}(2n)\) s.t. \(S^\top M S = D\oplus D\) with **symplectic eigenvalues** \(D=\mathrm{diag}(d_1,\dots,d_n)\). Map \(P\mapsto \tilde P = S^{-1}P\). The \(d_i\) themselves are **symplectic invariants** [Williamson1936], [Bhatia2018]. Residual symmetry \(\tilde S \in U(n)=\mathrm{SO}(2n)\cap\mathrm{Sp}(2n)\) can be addressed by rotation-invariant spectra or augmentation.
3. **Learned fine alignment (optional).** Predict a Hamiltonian \(H\in\mathfrak{sp}(2n)\) and set \(\hat S=\exp(H)\in\mathrm{Sp}(2n)\); train with an invariance loss so that encodings of \(P\) and \(\hat S P\) agree. The **exponential of a Hamiltonian matrix is symplectic** (standard Lie-group fact) [HamiltonianMatrixWiki].

**Augmentation.** Draw random \(H\) (small norm), set \(S=\exp(H)\), optionally add small translations, and enforce representation invariance across these transforms.

---

# 4. Architectures

- **Spherical CNN** on \(h_P\) or on a smoothed \(S_P\) field with rotation-invariant heads (power spectra or bispectra) [Cohen2018], [Esteves2018], [Kazhdan2003].
- **Set Transformer / Deep Sets** on \((u, h_P(u))\) samples or facet/vertex sets with multihead attention for duplicate/near-parallel detection [Zaheer2017], [Lee2019].
- **Graph encoders** on facet–vertex incidence when combinatorics helps; fuse late with spherical or set branches [GraphTransformerSurvey2024].

---

# 5. Training signals (“teacher distances”) & losses

- **Support-function distance:** \( \min_{S\in\mathrm{Sp}(2n)} \|h_{SP}-h_Q\|_{L^p(S^{2n-1})} \) computed on a sphere grid.  
- **OT on the sphere:** Distances between smoothed surface measures \(S_P,S_Q\) via **Sinkhorn OT** on the grid, or **sliced OT** adapted to \(S^{2}\) (extendable to \(S^{d-1}\)) [Cuturi2013], [PeyreCuturi2019], [Quellmalz2023,2024].  
- **MMD** between measure embeddings of \(S_P,S_Q\) (characteristic kernels on the sphere) [Sriperumbudur2010], [Gretton2012].

Use these as **contrastive/metric-learning teachers** (InfoNCE, triplet) to enforce that symplectomorphic polytopes/perturbations map to nearby codes, unrelated ones to distant codes.

**Caveat (EHZ capacity).** The Ekeland–Hofer–Zehnder capacity is a key symplectic invariant with combinatorial formulas for polytopes in certain settings [HaimKislev2017], but **computing the EHZ capacity is NP-hard** in general (even for simplices) [LeipoldVallentin2024AMS]. Use it sparingly as a supervised target or for small validation sets.

---

# 6. Decoders (optional)

- **H-rep decoder:** Predict offsets \(\hat b_k\) for a learned dictionary of normals \(\{n_k\}\); reconstruct \(\hat P=\{x: n_k^\top x \le \hat b_k\}\).  
- **V-rep decoder:** Predict a point cloud and take convex hull.  
In both cases, evaluate reconstruction with **support-loss** \( \sum_{u\in U} | h_{\hat P}(u)-h_P(u)| \) and a slow symplectic-alignment check.

---

# 7. Baselines to implement and evaluate

**Baseline 1 — Support↦Sphere.**  
Canonicalize (Williamson) \(\rightarrow\) sample \(h_P\) on Fibonacci/HEALPix grid \(\rightarrow\)
- (1a) Spherical CNN + rotation-invariant head; or  
- (1b) SH power spectra + MLP.  
Train with ASp-augmentation invariance + support/OT teacher.

**Baseline 2 — SurfaceMeasure↦Sphere.**  
Compute facet normals/areas \(\rightarrow\) smooth to a field on \(S^{2n-1}\) \(\rightarrow\) Spherical CNN (or kernel mean embedding + MLP). Train with MMD/OT teachers.

**Baseline 3 — Set Transformer.**  
Process \(\{(u, h_P(u))\}\) or \(\{(n_i,b_i)\}\) with Set Transformer. Train with same teachers and duplication/noise augmentations.

Optional **fusion**: late-fuse (concatenate) Baseline 1 with Baseline 3.

---

# 8. Evaluation protocol (what “good” looks like)

For each baseline, report:

1. **Invariance tests.** Apply random \(S=\exp(H)\in\mathrm{Sp}(2n)\) and translations \(t\). Measure \(\|E(P)-E(SP+t)\|\) (lower is better).  
2. **Continuity tests.** Small vertex/halfspace perturbations; addition/removal of near-duplicate elements; measure code drift vs. teacher distances.  
3. **Retrieval.** Given a query \(P\), retrieve nearest neighbors under teacher distance; compare to embedding-space neighbors (Recall@k, mAP).  
4. **Prediction tasks.** Regress \(\log \mathrm{vol}\), facet-count histograms, and (on 4D subsets) approximate capacities where feasible; report \(R^2\)/MAE.  
5. **Speed/memory.** Wall-clock for encoding vs. polytope size; memory footprint.  
6. **Ablations.** With/without Williamson canonicalization; with/without learned canonicalizer; Spherical CNN vs. SH spectra; grid resolution study.

---

# 9. Risks & mitigations

- **Residual \(U(n)\) ambiguity** after Williamson: use rotation-invariant spectra or augmentation.  
- **Noisy/degenerate polytopes** (nearly parallel facets): smooth measures, duplicate-suppression layers, robust sampling.  
- **Teacher costs** (OT): use coarser grids or sliced OT for training; reserve full Sinkhorn for validation.

---

# References (selected)

Convex geometry & support functions:  
- [Schneider2014] R. Schneider, *Convex Bodies: The Brunn–Minkowski Theory*, 2nd ed., Cambridge, 2014. ([Cambridge page](https://api.pageplace.de/preview/DT0400.9781107455009_A24880563/preview-9781107455009_A24880563.pdf))  
- [SupportFnWiki] “Support function (convex geometry),” Wikipedia (accessed 2025-10-12), states \(d_H(K,L)=\|h_K-h_L\|_\infty\) for convex bodies. <https://en.wikipedia.org/wiki/Support_function>  
- [Kousholt2015] A. Kousholt et al., *Reconstruction of convex bodies from surface tensors*, 2015 (notes), explicitly relating Hausdorff and support-function gaps. <https://pure.au.dk/ws/files/97725043/math_csgb_2015_10.pdf>  
- [Gonzalez2009] Á. González, *Measurement of areas on a sphere using Fibonacci lattices*, 2009. <https://arxiv.org/pdf/0912.4540>  
- [Gorski2005] K. M. Górski et al., *HEALPix*, *ApJ* 622, 2005. <https://ui.adsabs.harvard.edu/abs/2005ApJ...622..759G/abstract>  
- [SaffKuijlaars1997] E. B. Saff, A. B. J. Kuijlaars, *Distributing many points on a sphere*, 1997. <https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf>

Spherical encoders & rotation invariants:  
- [Cohen2018] T. S. Cohen et al., *Spherical CNNs*, ICLR/ICLR’18. <https://arxiv.org/abs/1801.10130>  
- [Esteves2018] C. Esteves et al., *Learning SO(3)-Equivariant Representations with Spherical CNNs*, ECCV’18. <https://openaccess.thecvf.com/content_ECCV_2018/papers/Carlos_Esteves_Learning_SO3_Equivariant_ECCV_2018_paper.pdf>  
- [Kazhdan2003] M. Kazhdan, T. Funkhouser, S. Rusinkiewicz, *Rotation invariant spherical harmonic representation of 3D shape*, 2003. <https://www.cs.jhu.edu/~misha/Papers/Kazhdan03.pdf>

Set/graph encoders:  
- [Zaheer2017] M. Zaheer et al., *Deep Sets*, NeurIPS’17. <https://arxiv.org/abs/1703.06114>  
- [Lee2019] J. Lee et al., *Set Transformer*, ICML’19. <https://arxiv.org/abs/1810.00825>  
- [Qi2017] C. R. Qi et al., *PointNet*, CVPR’17. <https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf>  
- [GraphTransformerSurvey2024] A. Shehzad et al., *Graph Transformers: A Survey*, 2024. <https://arxiv.org/abs/2407.09777>

Kernel mean embeddings & MMD:  
- [Sriperumbudur2010] B. K. Sriperumbudur et al., *Hilbert Space Embeddings and Metrics on Probability Measures*, JMLR 2010. <https://www.jmlr.org/papers/volume11/sriperumbudur10a/sriperumbudur10a.pdf>  
- [Muandet2017] K. Muandet et al., *Kernel Mean Embedding of Distributions: A Review and Beyond*, 2017. <https://www.nowpublishers.com/article/DownloadSummary/MAL-060>  
- [Gretton2012] A. Gretton et al., *A Kernel Two-Sample Test*, JMLR 2012. <https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf>

Optimal transport:  
- [Cuturi2013] M. Cuturi, *Sinkhorn Distances*, NeurIPS’13. <https://papers.neurips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf>  
- [PeyreCuturi2019] G. Peyré, M. Cuturi, *Computational Optimal Transport*, FnT ML 2019. <https://dl.acm.org/doi/10.1561/2200000073>  
- [Quellmalz2023] M. Quellmalz et al., *Sliced Optimal Transport on the Sphere*, 2023. <https://page.math.tu-berlin.de/~quellm/paper/QueBeiSte23.pdf>  
- [Quellmalz2024] M. Quellmalz et al., *Parallelly Sliced OT on Spheres and on SO(3)*, 2024. <https://link.springer.com/article/10.1007/s10851-024-01206-w>

Symplectic canonicalization & invariants:  
- [Williamson1936] J. Williamson, *On the Algebraic Problem Concerning the Normal Forms of Linear Dynamical Systems*, *Amer. J. Math.*, 1936. <https://www.jstor.org/stable/2371062>  
- [Bhatia2018] R. Bhatia, T. Jain, *On symplectic eigenvalues of positive definite matrices*, 2018. <https://arxiv.org/abs/1803.04647>  
- [HamiltonianMatrixWiki] “Hamiltonian matrix,” Wikipedia (accessed 2025-10-12): exponential of Hamiltonian is symplectic. <https://en.wikipedia.org/wiki/Hamiltonian_matrix>

Implicit representations & diff. optimization:  
- [Park2019] J. J. Park et al., *DeepSDF*, CVPR’19. <https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf>  
- [Mescheder2019] L. Mescheder et al., *Occupancy Networks*, CVPR’19. <https://arxiv.org/abs/1812.03828>  
- [Sitzmann2020] V. Sitzmann et al., *SIREN*, NeurIPS’20. <https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Paper.pdf>  
- [Agrawal2019] A. Agrawal et al., *Differentiable Convex Optimization Layers*, NeurIPS’19. <https://arxiv.org/abs/1910.12430>

Symplectic capacities (context):  
- [HaimKislev2017] P. Haim‑Kislev, *On the symplectic size of convex polytopes*, 2017. <https://ui.adsabs.harvard.edu/abs/2017arXiv171203494H/abstract>  
- [LeipoldVallentin2024AMS] K. Leipold, F. Vallentin, *Computing the EHZ Capacity is NP-hard*, *Proc. Amer. Math. Soc. Series B*, 2024. <https://www.ams.org/journals/bproc/2024-11-52/S2330-1511-2024-00250-6/viewer>
