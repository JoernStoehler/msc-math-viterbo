# Inbox Summary — Systolic Ratio (Ostrover reply)

- Date: 2025-10-15
- Source: Forwarded by Kai Cieliebak (paraphrased; no verbatim email text)
- Topic: Systolic ratio of convex polytopes; EHZ capacity; ML-based search
- Meeting note: 2025-10-13 (Mon) — Advisor aligned with project vision; acknowledged progress and next steps.

## Key Points

- No known prior work applying machine learning to study the (symplectic) systolic ratio of convex polytopes.
- It is plausible that polytopes exist with systolic ratio larger than those in the authors’ previous work; exploring methods to find them is worthwhile.
- EHZ capacity implementation (Haim-Kislev) exists in MATLAB for convex polytopes in R^{2n}; practical complexity grows roughly like F! (F = number of facets). Works quickly for small F (~10) but becomes impractically slow as F increases.
- Using that code with simple optimization (random starts + gradient descent) produced new examples with systolic ratio = 1, but none exceeding 1.
- Complexity note: Computing the EHZ capacity for simplices in 2n dimensions is NP-hard (Leipold–Vallentin, 2024).
- Related efforts:
  - Chaidez–Hutchings: alternative computation of EHZ in R^4 via action and Conley–Zehnder index; also examples with systolic ratio 1 (incl. the 24‑cell).
  - Rudolf–Krupp: algorithm for EHZ capacity of “Lagrangian products” of convex polytopes; public code available.
- Attached preprint mentions “neural networks”/“learning”: Optimization on Symplectic Embeddings (Gajewski, Goldin, Safin, Singh, Zhang; July 2019).

## Attachment Status

- File: `mail/Optimization_on_Symplectic_Embeddings (1).pdf`
- PDF metadata (extracted):
  - Title: Optimization on Symplectic Embeddings
  - Authors: Alexander Gajewski, Eli Goldin, Jakwanul Safin, Navtej Singh, Junhui Zhang
  - Date on title page: July 2019
- arXiv search (title/author queries): No exact match found. Treat as a private preprint under this title.

## References (for follow-up)

- Haim-Kislev (EHZ capacity for convex polytopes): https://link.springer.com/article/10.1007/s00039-019-00486-4
- MATLAB code (EHZ capacity): https://github.com/pazithaimkislev/EHZ-capacity
- NP-hardness for EHZ on simplices: https://arxiv.org/abs/2402.09914
- Chaidez–Hutchings (R^4 approach): https://www.aimsciences.org/article/doi/10.3934/jcd.2021016
- Rudolf–Krupp (Lagrangian products): https://link.springer.com/article/10.1007/s10711-022-00717-3
- Rudolf–Krupp code: https://github.com/S-Krupp/EHZ-capacity-of-polytopes

## Notes / Actions

- Continue exploration of polytopes with systolic ratio ≥ 1; aim to discover examples with ratio > 1.
- Given NP-hardness and factorial scaling, consider surrogate objectives, relaxations, or structure-exploiting cases for scalable search; keep exact EHZ capacity for verification on small instances.
- Optionally contact the “Optimization on Symplectic Embeddings” authors regarding status/permission to cite; do not redistribute without consent.
