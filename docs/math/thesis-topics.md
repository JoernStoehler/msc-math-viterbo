# Viterbo MSc Thesis Topic

This file tracks the single thesis topic for this repository. The official description is included verbatim below, followed by brief implementation notes.

---

Official MSc thesis description (verbatim)

## 2) Probing Viterbo’s conjecture

To each star-shaped (with respect to the origin) compact domain \(X \subset \mathbb{R}^{2n}\) with smooth boundary one can associate its **systolic ratio**
\[
\mathrm{sys}(X) := \frac{A_{\min}(X)^n}{n!\,\mathrm{vol}(X)}.
\]
Here \(\mathrm{vol}(X)\) is the volume of \(X\) and \(A_{\min}(X)\) is the minimal action \(A(\gamma)=\!\!\int_{\gamma}\!\lambda_{\mathrm{st}}\) of a closed Reeb orbit \(\gamma\) on \(\partial X\). This notion extends to the case that \(X\) has piecewise smooth boundary, see [1].

Viterbo’s conjecture asserts that \(\mathrm{sys}(X) \le 1\) if \(X\) is convex. It was disproved in [2] by producing an explicit polytope \(P \subset \mathbb{R}^4\) with \(\mathrm{sys}(P) > 1\). Nonetheless, it remains an interesting question to understand better how the systolic ratio behaves on the space of convex polytopes. For example, Viterbo’s conjecture for Lagrangian products of a centrally symmetric convex body in \(\mathbb{R}^n\) with its dual is equivalent to the famous Mahler conjecture in convex geometry, which is proved for \(n \le 3\) and open for \(n \ge 4\).

**Task.** Write an efficient computer program for computing the systolic ratio of polytopes in \(\mathbb{R}^4\), and use it to probe more precisely where and how Viterbo’s conjecture fails. One should also use machine learning to determine the systolic ratio more efficiently and predict regions in the landscape of polytopes where Viterbo’s conjecture is likely to hold or fail.

**Literature.**

[1] Julian Chaidez and Michael Hutchings, “Computing Reeb dynamics on 4d convex polytopes,” arXiv:2008.10111.

[2] Pazit Haim-Kislev and Yaron Ostrover, “A Counterexample to Viterbo’s Conjecture,” arXiv:2405.16513.

---

Implementation notes (non‑verbatim)

- Start with Chaidez–Hutchings to enumerate combinatorial Reeb orbits (fast pruning), then validate via Minkowski‑billiard formulations (Rudolf) or optimization pipelines (Krupp) as cross‑checks.
- Expect NP‑hard worst‑cases (Leipold–Vallentin); design heuristics (LP/SOCP relaxations, symmetry, warm‑starts, caching across local polytope moves).
- Maintain a unit‑test zoo (ellipsoids, polydisks, cubes, cross‑polytopes) for normalization/scaling sanity checks.
