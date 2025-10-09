---
status: source
updated: 2025-10-09
summary: Transcription of official topic 2 (Probing Viterbo’s Conjecture) from K. Cieliebak; topic 1 removed per project selection.
---

# Thesis Topic — Probing Viterbo’s Conjecture

Source: “Thesis topics” — K. Cieliebak (excerpt; topic 2 only).

To each star‑shaped (with respect to the origin) compact domain `X ⊂ R^{2n}` with smooth boundary, one can associate its systolic ratio

`sys(X) := A_min(X)^n / (n! · vol(X))`.

Here `vol(X)` is the volume of `X` and `A_min(X)` is the minimal action `A(γ) = ∮_γ λ_st` of a closed Reeb orbit `γ` on `∂X`. This notion extends to the case that `X` has piecewise smooth boundary; see [1].

Viterbo’s conjecture asserts that `sys(X) ≤ 1` if `X` is convex. It was disproved in [2] by producing an explicit polytope `P ⊂ R^4` with `sys(P) > 1`. Nonetheless, it remains an interesting question to understand better how the systolic ratio behaves on the space of convex polytopes. For example, Viterbo’s conjecture for Lagrangian products of a centrally symmetric convex body in `R^n` with its dual is equivalent to the famous Mahler conjecture in convex geometry, which is proved for `n ≤ 3` and open for `n ≥ 4`.

The task in this project is to write an efficient computer program for computing the systolic ratio of polytopes in `R^4`, and use it to probe more precisely where and how Viterbo’s conjecture fails. One should also use machine learning to determine the systolic ratio more efficiently and predict regions in the landscape of polytopes where Viterbo’s conjecture is likely to hold or fail.

## Literature

- [1] Julian Chaidez and Michael Hutchings, “Computing Reeb dynamics on 4d convex polytopes,” arXiv:2008.10111. https://arxiv.org/abs/2008.10111
- [2] Pazit Haim‑Kislev and Yaron Ostrover, “A Counterexample to Viterbo’s Conjecture,” arXiv:2405.16513. https://arxiv.org/abs/2405.16513
