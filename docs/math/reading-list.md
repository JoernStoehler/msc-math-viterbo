# Viterbo Conjecture — MSc Thesis Reading List

> **How to use this list.** Bullets are grouped (Essentials → Algorithms → Background/Context). Each entry has a **text citation**, a **clickable link**, a **what you’ll learn** blurb, and an **effort estimate**
> `⏱` in **relative units** only (1u ≈ a short skim; 2–3u ≈ careful read; 5–8u ≈ deep/technical). Priorities are marked with ⭐.

---

## A) Core statements, current status, and sharp connections

- ⭐ **Haim‑Kislev, Ostrover (2024). _A Counterexample to Viterbo’s Conjecture_.**  
  <https://arxiv.org/abs/2405.16513>  
  **What:** The paper that refutes Viterbo’s volume–capacity inequality via **Minkowski billiards**; shows capacities need not agree on convex domains. Clear statement, structure of the example, and implications.  
  **⏱ 3u (dense but focused).**

- ⭐ **Viterbo (2000). _Metric and Isoperimetric Problems in Symplectic Geometry_.**  
  Stable link (JAMS/JSTOR): <https://www.jstor.org/stable/2646224>  
  **What:** Original isoperimetric‑type framework for capacities and volume; sets the landscape for the conjecture. Read for precise normalizations and comparison axioms.  
  **⏱ 3u.**

- **Artstein‑Avidan, Karasev, Ostrover (2013/14). _From Symplectic Measurements to the Mahler Conjecture_.**  
  arXiv: <https://arxiv.org/abs/1303.4197> (PDF: <https://arxiv.org/pdf/1303.4197>)  
  **What:** Precise bridge between Viterbo‑type bounds and **Mahler’s volume product** for centrally symmetric bodies; crucial for interpreting product domains and duality.  
  **⏱ 2u.**

- **Artstein‑Avidan, Ostrover (2007/08). _A Brunn–Minkowski Inequality for Symplectic Capacities of Convex Domains_.**  
  arXiv: <https://arxiv.org/abs/0712.2631> (IMRN version: <https://academic.oup.com/imrn/article/doi/10.1093/imrn/rnn044/697105>)  
  **What:** Analogue of Brunn–Minkowski in the symplectic setting; techniques and corollaries you can re‑use when comparing families of polytopes.  
  **⏱ 2u.**

- **Ostrover (2014). _When Symplectic Topology Meets Banach Space Geometry_.**  
  PDF: <https://arxiv.org/pdf/1404.6954> (ICM‑style survey PDF mirror: <https://www.math.tau.ac.il/~ostrover/Research/ICM.pdf>)  
  **What:** Non‑technical overview of capacities, asymptotic versions of Viterbo’s inequality (up to constants), and links to convex geometry/Banach techniques.  
  **⏱ 2u (survey).**

- **Balitskiy (2015). _Equality cases in Viterbo’s conjecture and isoperimetric billiards_.**  
  arXiv: <https://arxiv.org/abs/1512.01657>  
  **What:** Billiard‑based equality cases (e.g., permutohedron × simplex), plus special‑case confirmations of the inequality. Good for intuition on when equality can/should happen.  
  **⏱ 1–2u.**

- **Vicente (2025). _The strong Viterbo conjecture and flavors of duality_.**  
  arXiv: <https://arxiv.org/abs/2505.07572>  
  **What:** Recent perspective on **normalized capacities coinciding** under dualities for Lagrangian products. Useful for mapping variants you might test computationally.  
  **⏱ 1–2u.**

---

## B) Algorithms & computation (polytopes, EHZ, Reeb dynamics, complexity)

- ⭐ **Chaidez, Hutchings (2020/21). _Computing Reeb dynamics on 4d convex polytopes_.**  
  arXiv: <https://arxiv.org/abs/2008.10111> (journal: <https://www.aimsciences.org/article/doi/10.3934/jcd.2021016>)  
  **What:** Defines **combinatorial Reeb orbits** on polytope boundaries; correspondence to smooth Reeb orbits preserving **action** and **CZ index**. Enables computer search for minimal action (**EHZ** in 4D). Experimental tests re: systolic ratios.  
  **⏱ 3u (methods + examples).**

- ⭐ **Leipold, Vallentin (2024). _Computing the EHZ capacity is NP‑hard_.**  
  arXiv: <https://arxiv.org/abs/2402.09914> (PDF: <https://arxiv.org/pdf/2402.09914>)  
  **What:** Proves NP‑hardness for polytopes (via reduction from Feedback Arc Set in bipartite tournaments). Sets expectations and motivates **heuristics** / structure‑exploiting algorithms.  
  **⏱ 2u.**

- **Rudolf (2022/24). _Minkowski Billiard Characterization of the EHZ‑capacity of Convex Lagrangian Products_.**  
  arXiv: <https://arxiv.org/abs/2203.01718> (journal page: <https://link.springer.com/article/10.1007/s10884-022-10228-0>)  
  **What:** General, rigorous equivalence between **c_EHZ(K×T)** and **shortest (K,T)‑Minkowski billiard** length—without smoothness/strict convexity. Gives a discrete path to compute c_EHZ for **polytopes**.  
  **⏱ 2–3u.**

- **Krupp (2020). _Calculating the EHZ Capacity of Polytopes_ (PhD).**  
  PDF: <https://kups.ub.uni-koeln.de/36196/1/DissertationKrupp.pdf>  
  **What:** A full **computational pipeline**: formulations (Abbondandolo–Majer max problem), LP/SOCP/SDP relaxations, and **code** (NumPy/MOSEK). Includes upper/lower bounds and runtime behavior on instances.  
  **⏱ 5u (reference).**

- **Code — Krupp et al.**  
  GitHub note in papers; see Springer article footnotes and thesis: <https://link.springer.com/articl
[... omitted 0 of 168 lines ...]

  **What:** Foundational multiplicity and stability results for closed characteristics on convex hypersurfaces; background for **why** \(A_{\min}\) exists and how indices behave.  
  **⏱ 4–5u (selective read).**

- **Wang (2009→2014). Stability/ellipticity results for closed characteristics.**  
  Survey link (EMS 2009 PDF): <https://ems.press/content/serial-article-files/31673> and J. Diff. Eq. (2014) example: <https://www.sciencedirect.com/science/article/pii/S0022123614002134>  
  **What:** Elliptic/irrationally elliptic cases and lower bounds on counts; helpful when analyzing the structure of action‑minimizing orbits on smoothed polytopes.  
  **⏱ 2u.**

---

## E) Nice‑to‑have context and indices

- **Cristofaro‑Gardiner (and collaborators).**  
  Example: JDG 2019 toric‑domains paper PDF: <https://projecteuclid.org/journals/journal-of-differential-geometry/volume-112/issue-2/Symplectic-embeddings-from-concave-toric-domains-into-convex-ones/10.4310/jdg/1559786421.pdf>  
  **What:** Symplectic embeddings in toric domains. While not directly EHZ, gives “feel” for capacity behavior across standard families (ellipsoids, polydisks).  
  **⏱ 1–2u.**

- **McDuff (2011). _The Hofer conjecture on embedding symplectic ellipsoids_.**  
  PDF: <https://arxiv.org/pdf/1008.1885>  
  **What:** Shows ECH capacities give sharp embedding obstructions for ellipsoids. Adds contrast to the EHZ world.  
  **⏱ 1–2u.**

- **Hutchings (2022). _An elementary alternative to ECH capacities_.**  
  PNAS: <https://www.pnas.org/doi/10.1073/pnas.2203090119>  
  **What:** Simplified capacities with similar strength to ECH in many 4D cases; skim for intuition and comparisons.  
  **⏱ 1u.**

---

## F) Quick order‑of‑reading (if you want a single pass)

1) Haim‑Kislev & Ostrover (2024) → 2) Viterbo (2000) → 3) A‑A/K/O (2013/14) → 4) Chaidez–Hutchings (2020) → 5) Leipold–Vallentin (2024) → 6) Rudolf (2022/24) → 7) Krupp (2020 + code) → 8) Brunn–Minkowski for capacities (2007/08) → 9) Long–Zhu (2002, selective) → 10) any ECH‑caps survey (Hutchings).

---

### Notes for implementation
- For **experiments on polytopes in \(\mathbb{R}^4\)**: start with Chaidez–Hutchings search for **combinatorial Reeb orbits** (fast pruning), then validate via Krupp/Rudolf billiard formulations; expect **NP‑hard** worst cases (Leipold–Vallentin), so build **heuristics** (LP/SOCP relaxations, symmetry reductions, warm‑starts, caching across local polytope moves).  
- For **products** \(K\times K^\circ\)**/**duality variants**: combine Artstein‑Avidan–Karasev–Ostrover with Shi/Vicente to choose promising families.  
- Keep a small **unit test zoo** (ellipsoids, polydisks, cubes, cross‑polytopes) to sanity‑check normalizations and scaling.
