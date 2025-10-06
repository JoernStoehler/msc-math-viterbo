# Viterbo Conjecture — MSc Thesis Reading List

> How to use this list. Bullets are grouped (Essentials → Algorithms → Background/Context). Each
> entry has a text citation, a clickable link, a “what you’ll learn” blurb, and an effort estimate
> ⏱ in relative units only (1u ≈ a short skim; 2–3u ≈ careful read; 5–8u ≈ deep/technical).
> Priorities are marked with ⭐.

This file was moved from `docs/math/reading-list.md` to keep docs flat and numbered.

<!-- The content below mirrors the original reading list; see git history for provenance. -->

## A) Core statements, current status, and sharp connections

- ⭐ Haim‑Kislev, Ostrover (2024). A Counterexample to Viterbo’s Conjecture.\
  https://arxiv.org/abs/2405.16513\ What: The paper that refutes Viterbo’s volume–capacity
  inequality via Minkowski billiards; shows capacities need not agree on convex domains. Clear
  statement, structure of the example, and implications.\
  ⏱ 3u (dense but focused).

- ⭐ Viterbo (2000). Metric and Isoperimetric Problems in Symplectic Geometry.\
  Stable link (JAMS/JSTOR): https://www.jstor.org/stable/2646224\ What: Original isoperimetric‑type
  framework for capacities and volume; sets the landscape for the conjecture. Read for precise
  normalizations and comparison axioms.\
  ⏱ 3u.

- Artstein‑Avidan, Karasev, Ostrover (2013/14). From Symplectic Measurements to the Mahler
  Conjecture.\
  arXiv: https://arxiv.org/abs/1303.4197 (PDF: https://arxiv.org/pdf/1303.4197)\ What: Bridge
  between Viterbo‑type bounds and Mahler’s volume product for centrally symmetric bodies.\
  ⏱ 2u.

- Artstein‑Avidan, Ostrover (2007/08). A Brunn–Minkowski Inequality for Symplectic Capacities of
  Convex Domains.\
  arXiv: https://arxiv.org/abs/0712.2631\ What: Analogue of Brunn–Minkowski in the symplectic
  setting.\
  ⏱ 2u.

- Ostrover (2014). When Symplectic Topology Meets Banach Space Geometry.\
  PDF: https://arxiv.org/pdf/1404.6954\ What: Survey of capacities, asymptotic inequalities, links
  to convex geometry.\
  ⏱ 2u.

- Balitskiy (2015). Equality cases in Viterbo’s conjecture and isoperimetric billiards.\
  arXiv: https://arxiv.org/abs/1512.01657\ What: Billiard‑based equality cases; special‑case
  confirmations.\
  ⏱ 1–2u.

- Vicente (2025). The strong Viterbo conjecture and flavors of duality.\
  arXiv: https://arxiv.org/abs/2505.07572\ What: Normalized capacities coinciding under dualities
  for Lagrangian products.\
  ⏱ 1–2u.

## B) Algorithms & computation (polytopes, EHZ, Reeb dynamics, complexity)

- ⭐ Chaidez, Hutchings (2020/21). Computing Reeb dynamics on 4d convex polytopes.\
  arXiv: https://arxiv.org/abs/2008.10111\ What: Combinatorial Reeb orbits on polytope boundaries;
  compute minimal action (EHZ in 4D).\
  ⏱ 3u.

- ⭐ Leipold, Vallentin (2024). Computing the EHZ capacity is NP‑hard.\
  arXiv: https://arxiv.org/abs/2402.09914\ What: NP‑hardness for EHZ on polytopes; motivates
  heuristics/relaxations.\
  ⏱ 2u.

- Rudolf (2022/24). Minkowski Billiard Characterization of the EHZ‑capacity of Convex Lagrangian
  Products.\
  arXiv: https://arxiv.org/abs/2203.01718\ What: Equivalence c_EHZ(K×T) ↔ shortest (K,T)‑Minkowski
  billiard; discrete route for polytopes.\
  ⏱ 2–3u.

- Krupp (2020). Calculating the EHZ Capacity of Polytopes (PhD).\
  PDF: https://kups.ub.uni-koeln.de/36196/1/DissertationKrupp.pdf\ What: Computational pipeline:
  formulations, LP/SOCP/SDP relaxations, and code.\
  ⏱ 5u.

## E) Nice‑to‑have context and indices

- Cristofaro‑Gardiner (and collaborators).\
  Example (JDG 2019):
  https://projecteuclid.org/journals/journal-of-differential-geometry/volume-112/issue-2/Symplectic-embeddings-from-concave-toric-domains-into-convex-ones/10.4310/jdg/1559786421.pdf\
  What: Symplectic embeddings in toric domains.\
  ⏱ 1–2u.

- McDuff (2011). The Hofer conjecture on embedding symplectic ellipsoids.\
  PDF: https://arxiv.org/pdf/1008.1885\ What: ECH capacities give sharp embedding obstructions for
  ellipsoids.\
  ⏱ 1–2u.

- Hutchings (2022). An elementary alternative to ECH capacities.\
  PNAS: https://www.pnas.org/doi/10.1073/pnas.2203090119\ What: Simplified capacities with similar
  strength to ECH in many 4D cases.\
  ⏱ 1u.

## F) Quick order‑of‑reading

1. Haim‑Kislev & Ostrover (2024) → 2) Viterbo (2000) → 3) A‑A/K/O (2013/14) → 4) Chaidez–Hutchings
   (2020) → 5) Leipold–Vallentin (2024) → 6) Rudolf (2022/24) → 7) Krupp (2020 + code) → 8)
   Brunn–Minkowski (2007/08) → 9) Long–Zhu (2002, selective) → 10) any ECH‑caps survey (Hutchings).

Notes for implementation (skim)

- For experiments on polytopes in R^4: start with Chaidez–Hutchings orbits, validate via
  Rudolf/Krupp; expect NP‑hard worst cases; use heuristics, symmetry and caching.
- For products K×K^∘ and dualities: combine Artstein‑Avidan–Karasev–Ostrover with Shi/Vicente to
  pick promising families.
- Maintain a small unit‑test zoo for normalization/scaling checks.
