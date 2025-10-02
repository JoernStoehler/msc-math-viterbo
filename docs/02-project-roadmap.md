# Roadmap

This roadmap balances literature work with incremental, testable Python code.

## Phase 1 — Foundations
- Collect precise definitions and properties of major capacities.
- Capture canonical examples (ellipsoids, polydisks, product bodies) for later computation.

## Phase 2 — Techniques & Results
- Summarize strongest partial results and proof ideas.
- Identify computationally tractable families for experiments.

## Phase 3 — Python Implementation & Experiments
- Implement core helpers for convex bodies and simple capacity proxies/estimates.
- Add unit tests for helpers, and examples as doctests or literate snippets.
- Optional: exploratory notebooks or scripts for volume vs. capacity comparisons.

### Active milestone — Polytope search utilities
- Finalize deterministic affine/Cartesian constructors so candidate families are easy to script.
- Implement Euclidean volume backends (reference + optimized) with cross-tests.
- Provide a systolic-ratio wrapper around ``c_EHZ`` and the new volume helpers.
- Stand up a reproducible search enumerator that combines the canonical catalog with random draws.
- Wire benchmarks/profilers for the new routines and extend pytest coverage to transformations and known counterexamples.

## Phase 4 — Synthesis
- Draft the survey with diagrams and a results map.
- Consolidate examples, benchmarks, and discussions of limitations.

Tracking: use GitHub Issues with the RFC template for larger design notes; smaller tasks as issues with clear acceptance criteria.
