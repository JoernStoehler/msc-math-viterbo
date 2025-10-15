---
title: "Future: Symplectic capacity milestone roadmap"
created: 2025-10-14
status: idea
owner: TBD
priority: high
labels: [future, math, datasets]
---

## Summary

Track the staged rollout for reliable symplectic capacity, volume, and cycle computations across increasingly complex polytopes. Each milestone unlocks broader experimentation and dataset scale; treat them as checkpoints when planning medium-term work.

## Milestones

1. **5×5 counterexample baseline**  
   Deliver correct capacity, volume, and minimal-action cycle for the 5×5 counterexample to Viterbo. Record reproducible values and smoke tests.
2. **4D simplex coverage**  
   Extend the baseline routines to any 4D simplex (both canonical and randomly generated), with validation against analytic values where available.
3. **Small polytope sweep**  
   Generalise the solver to arbitrary “small” polytopes (≤15 vertices or ≤10 facets), including Lagrangian products and random instances. Capture performance telemetry.
4. **Dataset-ready performance**  
   Optimise (Torch and/or C++ kernels) so the full small-polytope sweep completes fast enough to seed dataset generation jobs.
5. **Dataset production**  
   Produce a large, validated dataset of the above invariants. Document schema, audit checks, and distribution characteristics.
6. **Experiment velocity**  
   Streamline notebooks, scripts, and delegation briefs so new experiments (ideation → write-up) can ship via a handful of tasks with minimal orchestration overhead.

## Dependencies

- `src/viterbo/math/capacity_ehz/`
- `src/viterbo/math/capacity_ehz/` (EHZ solvers)
- `src/viterbo/math/volume.py`
- Dataset tooling under `src/viterbo/datasets/`
- Benchmarking hooks (`tests/performance/`)

## Notes

- Expect tight coupling with the ongoing 4D EHZ capacity and AtlasTiny upgrades; schedule milestone work only once those foundations are stable.
- Each milestone should land with explicit tests or validation scripts so the next stage has a trusted baseline.
