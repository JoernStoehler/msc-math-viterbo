# Thesis Topics (Notes)

Source: departmental notes adapted for this project. These are brief, implementation‑adjacent overviews rather than polished expositions.

---

## 1) Turing machines and dynamical systems

Basis articles:

- C. Moore, “Unpredictability and undecidability in dynamical systems,” Phys. Rev. Lett. 64 (1990), 2354–2357.
- C. Moore, “Generalized shifts: unpredictability and undecidability in dynamical systems,” Nonlinearity 4 (1991), 199–230.

Summary. Moore shows an equivalence between Turing machines (viewed as dynamical systems) and generalized shifts. By embedding generalized shifts into other systems, one obtains smooth dynamical systems that can simulate Turing machines. Undecidability of the halting problem then implies unpredictability stronger than chaos.

Context. Examples where Turing‑level unpredictability may occur include cellular automata, neural networks, gases of hard spheres, electrical circuits, and billiards; active research explores whether it occurs in the N‑body problem or in Euler/Navier–Stokes.

Additional references:

- R. Cardona, E. Miranda, D. Peralta‑Salas, F. Presas, “Constructing Turing complete Euler flows in dimension 3,” PNAS 118 (2021), e2026818118.
- R. Cardona, E. Miranda, D. Peralta‑Salas, “Rechnende Flüssigkeiten,” Spektrum der Wissenschaft, Feb. 2023.

Goal. Understand Moore’s articles and present them with the required background on Turing machines and dynamical systems.

---

## 2) Probing Viterbo’s conjecture (systolic ratio on polytopes)

For a star‑shaped compact domain X ⊂ ℝ²ⁿ with smooth boundary, define the systolic ratio

sys(X) := A_min(X)^n / (n! · vol(X)),

where vol(X) is the volume and A_min(X) is the minimal action A(γ)=∮_γ λ_st of a closed Reeb orbit γ on ∂X. This notion extends to piecewise‑smooth boundaries (polytopes).

Viterbo’s conjecture asserts sys(X) ≤ 1 for convex X. It was disproved by Haim‑Kislev and Ostrover (2024) via an explicit polytope P ⊂ ℝ⁴ with sys(P) > 1. It remains interesting to understand the behavior of the systolic ratio over the space of convex polytopes. For Lagrangian products of a centrally symmetric convex body K ⊂ ℝⁿ with its dual K^∘, the conjecture connects to Mahler’s volume product (proved for n ≤ 3, open for n ≥ 4).

Task. Implement an efficient program to compute/estimate the systolic ratio of polytopes in ℝ⁴ and use it to probe where/how Viterbo’s conjecture fails. Consider data‑driven (ML) helpers to predict regions where the conjecture likely holds or fails.

Literature:

- Julian Chaidez and Michael Hutchings, “Computing Reeb dynamics on 4d convex polytopes,” arXiv:2008.10111.
- Pazit Haim‑Kislev and Yaron Ostrover, “A Counterexample to Viterbo’s Conjecture,” arXiv:2405.16513.

Implementation notes.

- Start with Chaidez–Hutchings to enumerate combinatorial Reeb orbits (fast pruning), then validate via Minkowski‑billiard formulations (Rudolf) or optimization pipelines (Krupp) as cross‑checks.
- Expect NP‑hard worst‑cases (Leipold–Vallentin); design heuristics (LP/SOCP relaxations, symmetry, warm‑starts, caching across local polytope moves).
- Maintain a unit‑test zoo (ellipsoids, polydisks, cubes, cross‑polytopes) for normalization/scaling sanity checks.

