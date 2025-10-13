---
status: adopted
created: 2025-10-12
workflow: workflow
summary: Transcribe and contextualise the official MSc thesis topic on probing Viterbo’s conjecture.
---

# Workflow: Thesis Topic — Probing Viterbo’s Conjecture

## Context

- Source: “Thesis topics” by K. Cieliebak. We retain only Topic 2 (“Probing Viterbo’s Conjecture”); Topic 1 was dropped during project scoping.
- The topic targets star-shaped compact domains in $\mathbb{R}^{2n}$ with smooth (or piecewise smooth) boundary and focuses on the systolic ratio
  \[ \operatorname{sys}(X) = \frac{A_{\min}(X)^n}{n!\, \operatorname{vol}(X)}. \]
- Viterbo’s conjecture claims $\operatorname{sys}(X) \le 1$ for convex $X$; the counterexample of Haim-Kislev and Ostrover (2024) shows violations for suitable 4D polytopes.

## Objectives

- Implement efficient computation of the systolic ratio for 4D convex polytopes, leveraging Minkowski billiards and Reeb dynamics pipelines.
- Use optimisation and machine-learning heuristics to explore the landscape of convex polytopes, predicting where the conjecture holds or fails.
- Document literature context and dependencies so experimentation remains reproducible and theoretically grounded.

## Execution Outline

1. Build or reuse tooling under `src/viterbo/math/symplectic.py` and supporting helpers in `src/viterbo/math/geometry.py` to evaluate $A_{\min}(P)$ for convex polytopes.
2. Benchmark known counterexamples (e.g., the 2024 Haim-Kislev–Ostrover polytope) and equality cases to validate the implementation.
3. Explore parametric families (Lagrangian products, centrally symmetric bodies) to map regions where $\operatorname{sys}(P)$ crosses 1.
4. Train or calibrate ML surrogates that predict promising search directions; integrate with the `atlas_tiny` dataset scaffolding once completed or subsequent dataset presets when available.

## Literature

- Chaidez & Hutchings (2020/21): Computing Reeb dynamics on 4D convex polytopes. [arXiv:2008.10111](https://arxiv.org/abs/2008.10111)
- Haim-Kislev & Ostrover (2024): A Counterexample to Viterbo’s Conjecture. [arXiv:2405.16513](https://arxiv.org/abs/2405.16513)

## Acceptance

- Tooling for systolic ratio experiments aligns with the modern capacity/spectrum stacks and covers known counterexamples.
- Literature references and cross-links stay in sync with the reading list brief.

## Status Log

- 2025-02-14 — Migrated the legacy `docs/11-math-thesis-topics.md` transcription into the briefs tree with updated execution outline.
