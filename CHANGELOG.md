# Changelog

## 2025-10-27 — Systolic ratio normalization

- Normalization change for the systolic ratio (EHZ capacity + volume):
  - Previous: `Vol / c^n`
  - New: `c^n / (n! · Vol)`
- Rationale: Align with mainstream literature conventions so that the Euclidean ball in `R^{2n}` satisfies `Sys(B^{2n}) = 1`, and to match statements of volume–capacity inequalities and recent results.
- References:
  - C. Viterbo (1999), Functors and computations in Floer homology with applications, GAFA 9.
  - S. Artstein‑Avidan, R. Karasev, Y. Ostrover (2014), From Symplectic Measurements to the Mahler Conjecture, Duke Math. J.
  - P. Haim‑Kislev, Y. Ostrover (2024), A Counterexample to Viterbo’s Conjecture, arXiv:2405.16513.
- Affected code/tests/notebooks:
  - `src/viterbo/math/capacity_ehz/ratios.py` — `systolic_ratio(volume, capacity_ehz, dim)` implements `c^n / (n! · Vol)` and documents the change.
  - `tests/math/test_minimal_action.py` — verifies `Sys(B^{2n}) = 1` under the new convention in 2D/4D.
  - `notebooks/viterbo_counterexample_proposal.py` — computes systolic ratio using `c^n / (n! · Vol)` in 4D (n=2).
  - `notebooks/atlas_tiny_analysis.py` — consistency check updated to the new formula; comments reflect `c^n / (n! · Vol)`.
  - `src/viterbo/datasets/atlas_tiny.py` — dataset rows compute `systolic_ratio` via the shared formula; no call‑site changes required.
- Impact:
  - Euclidean ball: `Sys` remains 1 by definition of the convention.
  - Counterexample family (Haim–Kislev–Ostrover 2024): `Sys > 1` as reported in the paper; values now align numerically with literature.
  - Datasets: values are derived through the shared helper and update automatically when regenerating artefacts.

