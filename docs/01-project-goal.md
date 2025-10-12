# Project Goal and Context — Viterbo Conjecture

This project investigates the Viterbo conjecture, which proposes a sharp relationship between
symplectic capacity and volume for convex domains in symplectic Euclidean space. Roughly: among all
convex domains with fixed volume, the Euclidean ball maximizes fundamental symplectic capacities.
This connects symplectic topology with convex geometry and is closely related to symplectic
isoperimetric-type inequalities.

## Motivation

- Clarify the landscape of symplectic capacities and their extremal properties.
- Understand known partial results, special cases, and related inequalities.
- Explore computational or experimental approaches on convex bodies to build intuition and test
  conjectural behaviors.

## Guiding Questions

- How do major capacities (Gromov width, Hofer–Zehnder, Ekeland–Hofer, ECH capacities) compare on
  convex domains?
- What is precisely stated by the Viterbo conjecture and for which classes is it known/unknown?
- What are the strongest known partial results and their methods (e.g., generating functions,
  symplectic embeddings, convex-geometric techniques)?
- Are there quantitative or computational experiments on specific convex families (ellipsoids,
  polydisks, Lp-balls) that illuminate the conjecture?

## Proposed Deliverables

- A concise but thorough survey of capacities and the Viterbo conjecture, with clear statements,
  examples, and a map of implications between results.
- Worked examples computing or bounding capacities for standard convex bodies.
- Optional: small computational experiments exploring volume vs. capacity on parametric families of
  bodies.

## Modernization milestone and legacy sunset

- The production stack now lives entirely under :mod:`viterbo`; the legacy
  :mod:`viterbo.symplectic` modules have been removed as part of the modernization programme.
- Release messaging highlights that higher-dimensional (≥6D) capacity, cycle, and spectrum support
  remains on the backlog while we analyse combinatorial explosion and runtime budget implications.
- Downstream consumers should import `viterbo.capacity`, `viterbo.volume`, and related
  helpers directly; notebooks and scripts will be updated once the atlas pipeline lands.

## Roadmap (High-level)

1. Foundations: review capacities and symplectic embeddings; compile precise statements of the
   conjecture.
1. Results & Techniques: study partial results and special cases; emphasize proof strategies.
1. Examples: compute/bound capacities for standard convex bodies.
1. Synthesis: write survey with diagrams and a results map; optionally add computational
   experiments.

Documentation pointers:

- Reading list: `docs/12-math-reading-list.md`
- Current exploration programme: `docs/briefs/2025-10-07-task-systolic-overview.md`
- Thesis topic transcription placeholder: `docs/11-math-thesis-topics.md`
