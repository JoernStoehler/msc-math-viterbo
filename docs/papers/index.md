# Papers Index — MSc Math Viterbo

Purpose: one-stop list of all papers, preprints, notes, and ancillary sources we have considered so far. Entries link to local text sources under `docs/papers/` when available to avoid re-downloading. Keep this file append-only; move items between sections by updating their status rather than deleting.

Conventions

- For arXiv items, store text as `.tex` (preferred) or PDF-converted Markdown `paper.md` in a folder `docs/papers/YYYY-author-keyword/`.
- Include a short note on relevance and what we consult each item for.
- Status: `core`, `useful`, `background`, `tangential`, or `verify` (needs verification or metadata fix).
 - Takeaways: add brief, unbalanced bullet(s) capturing why we care. Some entries get a 5‑word gist; others 3–5 detailed bullets.
 - Use helper commands to fetch OA copies and index snippets:
   - `just fetch-arxiv ID=... STATUS=...`
   - `just fetch-doi DOI=... STATUS=...` (saves only if an OA PDF is found; otherwise prints a snippet you can paste with `Local: none`).

## Core Counterexample and Near-Ball Regime

- Haim-Kislev, Ostrover (2024) — “A counterexample to Viterbo's conjecture” — arXiv:2405.16513
  - Local: `docs/papers/2024-ostrover-counterexample-viterbo/` (LaTeX source)
  - Status: core — main counterexample; normalization checks and statements.
  - Takeaways:
    - First counterexample to Viterbo’s volume–capacity inequality using $c_{EHZ}$ on Lagrangian product of rotated regular pentagons (extends to all even dimensions via products).
    - Works with normalized capacities; shows normalized capacities need not coincide on convex domains.
    - Proof leverages explicit EHZ formulas for polytopes and Minkowski billiards; constants computed in closed form.

- Abbondandolo, Benedetti, Edtmair (2023) — “Symplectic capacities of domains close to the ball and Banach–Mazur geodesics…” — arXiv:2312.07363
  - Local: `docs/papers/2023-abbondandolo-benedetti-edtmair-near-ball/paper.md`
  - Status: core — all normalized capacities coincide near the ball.
  - Takeaways:
    - All normalized capacities coincide on $C^2$–near‑ball domains; failure at mere $C^1$ closeness.
    - Banach–Mazur geodesics viewpoint clarifies geometry of the near‑ball regime.

## Capacity Foundations, Conjecture, Bridges

- Viterbo (2000) — “Metric and isoperimetric problems in symplectic geometry” — JAMS
  - Local: `docs/papers/2000-viterbo-metric-isoperimetric/notes.md`
  - Status: core — original conjecture; no arXiv preprint found; journal access required.
  - Takeaways:
    - States volume–capacity isoperimetric inequality; ball extremizes normalized capacities at fixed volume.
    - Fixes normalization conventions (ball/cylinder) used by most later works.

- Balitskiy (2015) — “Equality cases in Viterbo-type inequalities” — preprint/article
  - Local: `docs/papers/2015-balitskiy-equality-cases-viterbo/permut.tex`
  - Status: useful — sharpness/equality structures around the conjecture.

- Cieliebak, Hofer, Latschev, Schlenk (2007) — “Quantitative symplectic geometry” — MSRI Publ. 54
  - Local: `docs/papers/2007-chls-quantitative-symplectic-geometry/paper.md` (preprint arXiv:math/0506191)
  - Status: useful — capacity axioms, normalization, early bounds.
  - Takeaways:
    - Canonical capacity axioms and normalizations; context for Gromov width and comparison principles.

- Artstein-Avidan, Milman, Ostrover (2008) — “The M-ellipsoid, symplectic capacities and volume” — CMH
  - Local: `docs/papers/2013-ostrover-symplectic-measurements-mahler/main.tex` (related line of work)
  - Status: useful — asymptotic volume–capacity relations.
  - Takeaways:
    - M‑ellipsoid machinery links convex geometry and capacities; asymptotic volume–capacity inequalities.

- Artstein-Avidan, Karasev, Ostrover (2014) — “From symplectic measurements to the Mahler conjecture” — Duke Math. J.
  - Local: `docs/papers/2013-ostrover-symplectic-measurements-mahler/main.tex`
  - Status: core — Mahler link; centrally symmetric regime.
  - Takeaways:
    - Equates Viterbo’s inequality (centrally symmetric bodies) with Mahler’s volume product; key conditional bridge.

- Ostrover (2014) — “When Symplectic Topology Meets Banach Space Geometry” — ICM Proceedings
  - Local: `docs/papers/2014-ostrover-when-symplectic-topology-meets-banach/ICM-Ostrover-final.tex`
  - Status: background — survey and normalization context.
  - Takeaways:
    - High‑level survey connecting symplectic capacities with asymptotic convex geometry and Banach‑space ideas.

- Irie (2019→2022) — “Symplectic homology of fiberwise convex sets and homology of loop spaces” — JSG 2022; arXiv:1907.09749
  - Local: `docs/papers/2019-irie-sh-fiberwise-convex-loop-spaces/paper.md`
  - Status: core — SH capacity and its relation to EHZ on convex bodies.
  - Takeaways:
    - Proves $c_{SH}(K)=c_{EHZ}(K)$ for convex bodies; loop‑space formula for $c_{SH}$ enables comparisons and subadditivity.

- Hutchings, Gutt, Ramos (2022) — “Examples around the strong Viterbo conjecture” — JFPTA; arXiv:2003.10854
  - Local: `docs/papers/2022-hutchings-gutt-ramos-strong-viterbo-examples/paper.md`
  - Status: useful — context and examples related to strong Viterbo.
  - Takeaways:
    - Curates examples testing stronger variants; provides constructions and obstructions relevant to capacity extremality.

- Vicente (2025) — “The strong Viterbo conjecture and various flavours of duality in Lagrangian products” — arXiv:2505.07572
  - Local: `docs/papers/2025-vicente-strong-viterbo-conjecture/`
  - Status: useful — variants and duality aspects post-counterexample.

## Computation and Algorithms (EHZ, Polytopes, Products)

- Haim-Kislev (2019) — “On the symplectic size of convex polytopes” — AGT
  - Local: `docs/papers/2019-haim-kislev-size-of-polytopes/paper.md` (preprint arXiv:1712.03494)
  - Status: core — EHZ capacity algorithm on polytopes.
  - Takeaways:
    - Combinatorial formula/algorithm for $c_{EHZ}$ on polytopes; useful for certified small‑F cases.
    - Subadditivity and structural properties; factorial‑like growth hints at complexity barriers.

- Chaidez, Hutchings (2020–2021) — “Computing Reeb dynamics on 4d convex polytopes” — arXiv:2008.10111; journal version later
  - Local: `docs/papers/2020-hutchings-computing-reeb-polytopes/`
  - Status: useful — 4D specialization via CZ index.
  - Takeaways:
    - Combinatorial Reeb orbits on smoothed 4D polytopes correspond to Reeb orbits with matched action/CZ index; enables computations.

- Rudolf, Krupp (2022) — “EHZ capacity for Lagrangian products of convex polytopes” — Geom. Dedicata
  - Local: `docs/papers/2022-rudolf-minkowski-billiard-ehz/Published_Article_for_Arxiv.tex`
  - Status: core — product cases; billiard link.
  - Takeaways:
    - Formulae reducing $c_{EHZ}(K\times T)$ to shortest Minkowski billiards; efficient for product‑structured shapes.

- Haim-Kislev, Ostrover (2021) — “Remarks on symplectic capacities of p-products” — arXiv:2111.09177
  - Local: `docs/papers/2021-haim-kislev-ostrover-symplectic-p-products/paper.md`
  - Status: useful — capacity behavior under $p$‑products.
  - Takeaways:
    - Tensor‑power trick for asymptotic variants; subadditivity and behavior of capacities under $p$‑products.

- Krupp (2020) — PhD thesis on EHZ capacity
  - Local: `docs/papers/2020-krupp-ehz-capacity-phd/notes.md`
  - Status: background — extended exposition.

## Billiards and Minkowski/Finsler Background

- Gutkin, Tabachnikov (2002) — “Billiards in Finsler and Minkowski geometries” — JGP
  - Local: none (no arXiv preprint found; use journal/author sites if OA copy appears)
  - Status: background — Minkowski billiards framework.
  - Takeaways:
    - Foundational Minkowski/Finsler billiards reference for translating capacity problems to billiard trajectories.

- Akopyan, Karasev (2022; preprint 2019) — “When different norms lead to same billiard trajectories?” — EJM; arXiv:1912.12685
  - Local: `docs/papers/2019-akopyan-karasev-billiard-norms/paper.md`
  - Status: useful — norm equivalence phenomena; relates to billiard trajectories.
  - Takeaways:
    - Norm equivalence implications for billiard trajectories; informs when billiard‑based capacity reductions are robust.

## ECH Capacities and 4D Context

- Cristofaro-Gardiner, Hutchings, Ramos (2015) — “The asymptotics of ECH capacities” — Invent. Math.
  - Local: `docs/papers/2015-cristofaro-gardiner-hutchings-ramos-ech-asymptotics/paper.md` (preprint arXiv:1210.2167)
  - Status: useful — 4D capacity asymptotics.
  - Takeaways:
    - ECH capacities recover volume asymptotically on Liouville domains; benchmark for 4D capacity comparisons.

- Cristofaro-Gardiner, Savale (2020) — “Sub-leading asymptotics of ECH capacities” — Selecta Math.
  - Local: `docs/papers/2020-cristofaro-gardiner-savale-subleading-ech/paper.md` (preprint arXiv:1811.00485)
  - Status: useful — refinements in 4D.
  - Takeaways:
    - Improves bounds on sub‑leading ECH terms; sharper quantitative control for embedding/inequality problems.

- Hutchings (2014) — Lecture notes on ECH
  - Local: none yet (author notes likely OA on personal page; to fetch)
  - Status: background — reference notes.
  - Takeaways:
    - Primer on ECH foundations; definitions, action filtration, and index.

- Hutchings (2022) — “An elementary alternative to ECH capacities” — PNAS; arXiv:2201.03143
  - Local: `docs/papers/2022-hutchings-elementary-ech-capacities/proceedings.tex`
  - Status: useful — alternate formulation.
  - Takeaways:
    - Simplified alternative to ECH capacities; practical for some computations and expository clarity.

- Wormleighton (2021) — “ECH capacities, Ehrhart theory, and toric varieties” — JSG; arXiv:1906.02237
  - Local: `docs/papers/2021-wormleighton-ech-ehrhart-toric/paper.md`
  - Status: background — combinatorial aspects.
  - Takeaways:
    - Relates ECH capacities of convex toric domains to Ehrhart theory via a polarized toric variety construction.
    - Gives asymptotic/combinatorial insights that can aid capacity computations on toric domains.

- Hutchings et al. (2020) — “Computing Reeb dynamics on polytopes” (working title in notes)
  - Local: `docs/papers/2020-hutchings-computing-reeb-polytopes/`
  - Status: background — Reeb dynamics on polytopes; context for EHZ on polytopes.

## Embeddings (Context)

- McDuff, Schlenk (2012) — “The embedding capacity of 4-dimensional symplectic ellipsoids” — Annals of Math.
  - Local: `docs/papers/2011-mcduff-hofer-conjecture-ellipsoids/hofercnov30finxxx.tex` (related notes)
  - Status: useful — embedding thresholds; context for capacities.
  - Takeaways:
    - Sharp ellipsoid‑into‑ball embeddings in 4D; classical constraints shaping capacity comparisons.

- Schlenk (1999) — “On symplectic folding” — arXiv:math/9903086
  - Local: `docs/papers/1999-schlenk-symplectic-folding/paper.md`
  - Status: background — embedding techniques.
  - Takeaways:
    - Folding constructions as flexible embedding tools; complements rigidity‑based bounds.

- Hind (2014) — “Some optimal embeddings of symplectic ellipsoids” — arXiv:1409.5110
  - Local: `docs/papers/2014-hind-optimal-ellipsoid-embeddings/paper.md`
  - Status: background — explicit embeddings.
  - Takeaways:
    - Concrete high‑dimensional embeddings; useful reference points beyond 4D.

- Hutchings et al. (2024) — “Anchored symplectic embeddings” — arXiv:2407.08512
  - Local: `docs/papers/2024-hutchings-anchored-embeddings/paper.md`
  - Status: background — constraints and quantitative criteria.
  - Takeaways:
    - Anchoring constraints alter embedding feasibility; highlights sensitivity to boundary conditions.

- Cristofaro-Gardiner (2019) — symplectic embeddings survey/context
  - Local: `docs/papers/2019-cristofaro-gardiner-symplectic-embeddings/notes.md`
  - Status: background — survey-style context for embeddings.

- Hutchings (2011) — “Recent progress on symplectic embedding problems in four dimensions” — PNAS; arXiv:1101.1069
  - Local: `docs/papers/2011-hutchings-embeddings-survey-pnas/paper.md`
  - Status: background — concise 4D embedding survey.
  - Takeaways:
    - Summarizes 4D embedding landscape; useful quick reference for capacity/embedding interplay.

## Complexity and Normalization Variants

- Leipold, Vallentin (2024) — “Computing the EHZ capacity of simplices is NP-hard” — arXiv:2402.09914
  - Local: `docs/papers/2024-vallentin-ehz-np-hard/main.tex`
  - Status: core — complexity limits.
  - Takeaways:
    - EHZ computation for simplices is NP‑hard; supports focus on special cases/approximations.

- Altabar, et al. (2022) — “Cube normalized symplectic capacities” — arXiv:2208.13666
  - Local: `docs/papers/2022-cube-normalized-capacities/paper.md`
  - Status: useful — alternate normalization; compare to ball/cylinder.
  - Takeaways:
    - Alternative normalization calibrated to the cube; useful foil for standard ball/cylinder normalizations.

## Toric/Delzant and Related

- Ostrover, Tyomkin (2009) — “On the quantum homology algebra of toric Fano manifolds” — Selecta Math.; arXiv:0804.0270
  - Local: `docs/papers/2009-ostrover-tyomkin-quantum-homology-toric-fano/paper.md`
  - Status: background — toric context; capacities indirectly.
  - Takeaways:
    - Criteria for semi-simplicity/field summands in quantum homology of toric Fano manifolds; context for toric invariants.

- Ostrover (2007) — “Brunn–Minkowski type inequalities in symplectic geometry” (or related) — notes/early preprint
  - Local: `docs/papers/2007-ostrover-brunn-minkowski-symplectic/main.tex`
  - Status: background — connections to convex-geometry inequalities.
  - Takeaways:
    - Bridges symplectic and convex isoperimetry; background for volume‑type inequalities.

## Foundational

- Gromov (1985) — “Pseudoholomorphic curves in symplectic manifolds” — Invent. Math.
  - Local: none yet
  - Status: background — Gromov width; holomorphic curves.
  - Takeaways:
    - Foundational source for pseudoholomorphic curves and Gromov width; baseline rigidity tool.

## Notes/Briefs and Non-Archival Sources Considered

- Gil Kalai (2024) — Blog post: “Viterbo’s conjecture was refuted …”
  - Local: none
  - Status: background — context; link to primary sources.

- Various lecture notes and slides (Karshon capacities notes 2006–07; seminar slides; Reeb dynamics notes)
  - Local: none
  - Status: background — supplemental expository material.
