---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Curated reading plan for the Viterbo conjecture project with priorities and effort estimates.
---

# Workflow: Viterbo Conjecture Reading List

## Context

- Bullets are grouped into thematic tracks with priorities (⭐) and relative effort units (1u ≈ skim, 3u ≈ deep read, 5–8u ≈ very technical).
- The list mirrors the structure originally kept in `docs/12-math-reading-list.md`; this brief keeps the content in the modern tree.
- See [2025-10-12-workflow-reading-archive.md](2025-10-12-workflow-reading-archive.md) for mirrored LaTeX sources and metadata notes.

## Track A — Core statements, status, and dualities

- ⭐ Haim-Kislev, Ostrover (2024). *A Counterexample to Viterbo’s Conjecture.* https://arxiv.org/abs/2405.16513 — Minkowski billiards refute the conjectured bound via a 4D polytope example. ⏱ 3u.
- ⭐ Viterbo (2000). *Metric and Isoperimetric Problems in Symplectic Geometry.* https://www.jstor.org/stable/2646224 — Original framework for capacities, volume, and comparison axioms. ⏱ 3u.
- Artstein-Avidan, Karasev, Ostrover (2013/14). *From Symplectic Measurements to the Mahler Conjecture.* https://arxiv.org/abs/1303.4197 — Relates Viterbo-type bounds to Mahler’s volume product. ⏱ 2u.
- Artstein-Avidan, Ostrover (2007/08). *A Brunn–Minkowski Inequality for Symplectic Capacities of Convex Domains.* https://arxiv.org/abs/0712.2631 — Brunn–Minkowski analogue for capacities. ⏱ 2u.
- Ostrover (2014). *When Symplectic Topology Meets Banach Space Geometry.* https://arxiv.org/pdf/1404.6954 — Survey connecting capacities to convex geometry. ⏱ 2u.
- Balitskiy (2015). *Equality cases in Viterbo’s conjecture and isoperimetric billiards.* https://arxiv.org/abs/1512.01657 — Equality cases via billiard trajectories. ⏱ 1–2u.
- Vicente (2025). *The strong Viterbo conjecture and flavors of duality.* https://arxiv.org/abs/2505.07572 — Normalisation coincidences under dualities for Lagrangian products. ⏱ 1–2u.

## Track B — Algorithms & computation (polytopes, EHZ, Reeb dynamics, complexity)

- ⭐ Chaidez, Hutchings (2020/21). *Computing Reeb dynamics on 4D convex polytopes.* https://arxiv.org/abs/2008.10111 — Combinatorial algorithm for closed Reeb orbits on polytope boundaries. ⏱ 3u.
- ⭐ Leipold, Vallentin (2024). *Computing the EHZ capacity is NP-hard.* https://arxiv.org/abs/2402.09914 — NP-hardness reduction motivating relaxations and heuristics. ⏱ 2u.
- Rudolf (2022/24). *Minkowski Billiard Characterization of the EHZ-capacity of Convex Lagrangian Products.* https://arxiv.org/abs/2203.01718 — Shortest $(K,T)$-Minkowski billiards compute $c_{\mathrm{EHZ}}(K \times T)$. ⏱ 2–3u.
- Krupp (2020). *Calculating the EHZ Capacity of Polytopes.* https://kups.ub.uni-koeln.de/36196/1/DissertationKrupp.pdf — Optimisation models (LP/SOCP/SDP) and practical code guidance. ⏱ 5u.

## Track E — Context and indices

- Cristofaro-Gardiner *et al.* (2019). *Symplectic embeddings from concave toric domains into convex ones.* https://projecteuclid.org/.../10.4310/jdg/1559786421.pdf — Embedding obstructions via ECH capacities. ⏱ 1–2u.
- McDuff (2011). *The Hofer conjecture on embedding symplectic ellipsoids.* https://arxiv.org/pdf/1008.1885 — ECH capacities settle the Hofer conjecture for 4D ellipsoids. ⏱ 1–2u.
- Hutchings (2022). *An elementary alternative to ECH capacities.* https://www.pnas.org/doi/10.1073/pnas.2203090119 — Combinatorial capacities approximating ECH obstructions. ⏱ 1u.

## Suggested order of reading

1. Haim-Kislev & Ostrover (2024)
2. Viterbo (2000)
3. Artstein-Avidan, Karasev, Ostrover (2013/14)
4. Chaidez–Hutchings (2020/21)
5. Leipold–Vallentin (2024)
6. Rudolf (2022/24)
7. Krupp (2020)
8. Brunn–Minkowski (2007/08)
9. Long–Zhu (2002, selective)
10. Any ECH capacity survey (e.g., Hutchings)

## Implementation notes

- For 4D polytope experiments, start with Chaidez–Hutchings orbits, validate via Rudolf/Krupp, and expect NP-hard worst cases; lean on symmetry and caching.
- For Lagrangian products $K \times K^{\circ}$, combine Artstein-Avidan–Karasev–Ostrover with Shi/Vicente to identify promising families.
- Maintain regression tests for normalisation and scaling as algorithms evolve.

## Status Log

- 2025-02-14 — Migrated reading list from `docs/12-math-reading-list.md`; linked to the reading archive brief and preserved effort estimates.
