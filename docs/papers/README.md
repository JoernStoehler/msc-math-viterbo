# Symplectic Geometry Reading Archive

This folder mirrors the thesis reading list in `docs/12-math-reading-list.md`. For each paper we either store the LaTeX source obtained from arXiv or (when only a PDF is available) a metadata note that points to the stable download location. Binary assets were removed to keep the repository text-only.

| Year | Authors (lead) | Title | Link | Repository path | Summary |
| --- | --- | --- | --- | --- | --- |
| 2025 | Vicente | The strong Viterbo conjecture and flavors of duality | [arXiv:2505.07572](https://arxiv.org/abs/2505.07572) | `2025-vicente-strong-viterbo-conjecture/` | Normalizes EHZ- and Hofer-type capacities on Lagrangian products and shows coincidence under Legendre duality heuristics for centrally symmetric bodies. |
| 2024 | Haim-Kislev, Ostrover | A Counterexample to Viterbo’s Conjecture | [arXiv:2405.16513](https://arxiv.org/abs/2405.16513) | `2024-ostrover-counterexample-viterbo/` | Constructs a Minkowski billiard domain where EHZ and volume normalizations contradict Viterbo’s conjectured bound, highlighting polytope billiards. |
| 2024 | Leipold, Vallentin | Computing the EHZ capacity is NP-hard | [arXiv:2402.09914](https://arxiv.org/abs/2402.09914) | `2024-vallentin-ehz-np-hard/` | Reduces from partition-type problems to prove NP-hardness of EHZ capacity on 4d polytopes, motivating approximation schemes. |
| 2022 | Rudolf | Minkowski Billiard Characterization of the EHZ-capacity of Convex Lagrangian Products | [arXiv:2203.01718](https://arxiv.org/abs/2203.01718) | `2022-rudolf-minkowski-billiard-ehz/` | Shows shortest (K,T)-Minkowski billiards compute c_EHZ(K×T), enabling discrete search approaches. |
| 2022 | Hutchings | An elementary alternative to ECH capacities | [arXiv:2112.15027](https://arxiv.org/abs/2112.15027) | `2022-hutchings-elementary-ech-capacities/` | Introduces combinatorial capacities approximating ECH obstructions, easier to evaluate for 4d toric domains. |
| 2021 | Chaidez, Hutchings | Computing Reeb dynamics on 4d convex polytopes | [arXiv:2008.10111](https://arxiv.org/abs/2008.10111) | `2020-hutchings-computing-reeb-polytopes/` | Provides a combinatorial algorithm for Reeb orbits on polytope boundaries to evaluate EHZ via action minimization. |
| 2020 | Krupp | Calculating the EHZ Capacity of Polytopes (PhD thesis) | [Uni Köln](https://kups.ub.uni-koeln.de/36196/) | `2020-krupp-ehz-capacity-phd/` | Thesis with detailed optimization models (LP/SOCP/SDP) and code guidance for practical computation. |
| 2015 | Balitskiy | Equality cases in Viterbo’s conjecture and isoperimetric billiards | [arXiv:1512.01657](https://arxiv.org/abs/1512.01657) | `2015-balitskiy-equality-cases-viterbo/` | Classifies equality cases via billiard trajectories, especially simplices and Hanner polytopes. |
| 2014 | Ostrover | When Symplectic Topology Meets Banach Space Geometry | [arXiv:1404.6954](https://arxiv.org/abs/1404.6954) | `2014-ostrover-when-symplectic-topology-meets-banach/` | ICM survey tying symplectic capacities to convex geometry, including Mahler conjecture links. |
| 2013 | Artstein-Avidan, Karasev, Ostrover | From Symplectic Measurements to the Mahler Conjecture | [arXiv:1303.4197](https://arxiv.org/abs/1303.4197) | `2013-ostrover-symplectic-measurements-mahler/` | Relates symplectic capacities to Mahler’s volume product, showing equivalences for centrally symmetric bodies. |
| 2011 | McDuff | The Hofer conjecture on embedding symplectic ellipsoids | [arXiv:1008.1885](https://arxiv.org/abs/1008.1885) | `2011-mcduff-hofer-conjecture-ellipsoids/` | Uses ECH capacities to settle the Hofer conjecture for 4d ellipsoids, highlighting capacity sequences. |
| 2007 | Artstein-Avidan, Ostrover | A Brunn–Minkowski inequality for symplectic capacities of convex domains | [arXiv:0712.2631](https://arxiv.org/abs/0712.2631) | `2007-ostrover-brunn-minkowski-symplectic/` | Establishes a Brunn–Minkowski-style inequality for symplectic capacities of convex bodies. |
| 2000 | Viterbo | Metric and Isoperimetric Problems in Symplectic Geometry | [JAMS/JSTOR](https://www.jstor.org/stable/2646224) | `2000-viterbo-metric-isoperimetric/` | Foundational statement of the Viterbo conjecture; metadata only because the journal PDF is paywalled. |
| 2019 | Cristofaro-Gardiner *et al.* | Symplectic embeddings from concave toric domains into convex ones | [Project Euclid](https://projecteuclid.org/journals/journal-of-differential-geometry/volume-112/issue-2/Symplectic-embeddings-from-concave-toric-domains-into-convex-ones/10.4310/jdg/1559786421.pdf) | `2019-cristofaro-gardiner-symplectic-embeddings/` | Provides embedding obstructions and constructions via ECH capacities; metadata only (PDF download required). |

## Usage notes

- ArXiv directories contain the original LaTeX sources (with PDFs and image binaries removed). The files can be compiled locally if the missing figures are restored from the original downloads.
- Metadata-only directories document sources that could not be mirrored in text form because only paywalled or binary PDFs are available.
- For a quick reading order, see Section F of `docs/12-math-reading-list.md`.
