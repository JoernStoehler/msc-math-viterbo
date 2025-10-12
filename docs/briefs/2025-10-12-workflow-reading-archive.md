---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Index of mirrored paper sources and metadata for the symplectic geometry reading archive.
---

# Workflow: Symplectic Geometry Reading Archive

## Context

- Directory `docs/papers/` mirrors sources referenced in the reading list. When arXiv provides LaTeX, we keep text-only versions (PDFs and binaries stripped); otherwise we store metadata notes.
- This brief replaces the legacy `docs/papers/README.md` and keeps the navigation in the briefs tree.

## Structure

| Year | Lead authors | Title | Link | Repository path | Notes |
| ---- | ------------- | ----- | ---- | --------------- | ----- |
| 2025 | Vicente | *The strong Viterbo conjecture and flavors of duality* | [arXiv:2505.07572](https://arxiv.org/abs/2505.07572) | `docs/papers/2025-vicente-strong-viterbo-conjecture/` | Normalises Hofer- and EHZ-type capacities on Lagrangian products via duality arguments. |
| 2024 | Haim-Kislev, Ostrover | *A Counterexample to Viterbo’s Conjecture* | [arXiv:2405.16513](https://arxiv.org/abs/2405.16513) | `docs/papers/2024-ostrover-counterexample-viterbo/` | Minkowski billiards produce a 4D counterexample; mirrors appear in experiments. |
| 2024 | Leipold, Vallentin | *Computing the EHZ capacity is NP-hard* | [arXiv:2402.09914](https://arxiv.org/abs/2402.09914) | `docs/papers/2024-vallentin-ehz-np-hard/` | NP-hardness reduction guiding optimisation formulations. |
| 2022 | Rudolf | *Minkowski Billiard Characterization of the EHZ-capacity of Convex Lagrangian Products* | [arXiv:2203.01718](https://arxiv.org/abs/2203.01718) | `docs/papers/2022-rudolf-minkowski-billiard-ehz/` | Discrete $(K,T)$-billiard characterisation for product domains. |
| 2022 | Hutchings | *An elementary alternative to ECH capacities* | [arXiv:2112.15027](https://arxiv.org/abs/2112.15027) | `docs/papers/2022-hutchings-elementary-ech-capacities/` | Combinatorial capacities approximating ECH obstructions. |
| 2021 | Chaidez, Hutchings | *Computing Reeb dynamics on 4d convex polytopes* | [arXiv:2008.10111](https://arxiv.org/abs/2008.10111) | `docs/papers/2020-hutchings-computing-reeb-polytopes/` | Graph-based enumeration for Reeb dynamics, reused in tooling. |
| 2020 | Krupp | *Calculating the EHZ Capacity of Polytopes* | [Uni Köln](https://kups.ub.uni-koeln.de/36196/) | `docs/papers/2020-krupp-ehz-capacity-phd/` | PhD thesis with optimisation models and code guidance. |
| 2015 | Balitskiy | *Equality cases in Viterbo’s conjecture and isoperimetric billiards* | [arXiv:1512.01657](https://arxiv.org/abs/1512.01657) | `docs/papers/2015-balitskiy-equality-cases-viterbo/` | Equality cases and billiard trajectories. |
| 2014 | Ostrover | *When Symplectic Topology Meets Banach Space Geometry* | [arXiv:1404.6954](https://arxiv.org/abs/1404.6954) | `docs/papers/2014-ostrover-when-symplectic-topology-meets-banach/` | Survey tying capacities to convex geometry. |
| 2013 | Artstein-Avidan, Karasev, Ostrover | *From Symplectic Measurements to the Mahler Conjecture* | [arXiv:1303.4197](https://arxiv.org/abs/1303.4197) | `docs/papers/2013-ostrover-symplectic-measurements-mahler/` | Relates capacity inequalities to Mahler’s volume product. |
| 2011 | McDuff | *The Hofer conjecture on embedding symplectic ellipsoids* | [arXiv:1008.1885](https://arxiv.org/abs/1008.1885) | `docs/papers/2011-mcduff-hofer-conjecture-ellipsoids/` | ECH capacities settle the Hofer embedding conjecture for ellipsoids. |
| 2007 | Artstein-Avidan, Ostrover | *A Brunn–Minkowski inequality for symplectic capacities of convex domains* | [arXiv:0712.2631](https://arxiv.org/abs/0712.2631) | `docs/papers/2007-ostrover-brunn-minkowski-symplectic/` | Brunn–Minkowski analogue for capacities. |
| 2000 | Viterbo | *Metric and Isoperimetric Problems in Symplectic Geometry* | [JSTOR](https://www.jstor.org/stable/2646224) | `docs/papers/2000-viterbo-metric-isoperimetric/` | Metadata only because the journal PDF is paywalled. |
| 2019 | Cristofaro-Gardiner *et al.* | *Symplectic embeddings from concave toric domains into convex ones* | [Project Euclid](https://projecteuclid.org/journals/journal-of-differential-geometry/volume-112/issue-2/Symplectic-embeddings-from-concave-toric-domains-into-convex-ones/10.4310/jdg/1559786421.pdf) | `docs/papers/2019-cristofaro-gardiner-symplectic-embeddings/` | Metadata only (PDF download required). |

## Usage notes

- ArXiv directories contain original LaTeX sources with binaries removed; rehydrate locally if figures are needed.
- Metadata-only directories document sources that cannot be mirrored in text form.
- For reading order guidance, consult [2025-10-12-workflow-reading-list.md](2025-10-12-workflow-reading-list.md).

## Status Log

- 2025-02-14 — Migrated the `docs/papers/README.md` index into the briefs tree and refreshed cross-links.
