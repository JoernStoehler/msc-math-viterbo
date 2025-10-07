---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Pursue optimisation, analysis, and literature sweep directions extending the core systolic program.
---

# Advanced optimization and analysis directions

## Motivation

The ChatGPT Pro outline closes with supplementary tracks—shape derivatives, optimisation, topological data analysis, dynamical convexity, and literature monitoring—that refine or extend the main program.

## Scope

- Implement shape-derivative calculations for \(c_{\mathrm{EHZ}}\) via support number perturbations, providing gradients for search.
- Run optimisation loops (CMA-ES, Bayesian optimisation) over polytope support numbers, handing promising violators to [Rigorous certificates](2025-10-07-task-systolic-certificates.md).
- Apply topological data analysis to action spectra, study dynamical convexity of smoothings, and maintain a literature log on new systolic bounds.

## Deliverables

1. Gradient/shape derivative tooling integrated with the systolic pipeline.
2. Optimisation workflows with clear seeding, constraints, and certificate hand-offs.
3. Reports summarising TDA findings, dynamical convexity status, and literature updates.

## Dependencies & Links

- Builds on [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md) and data from [Landscape scans](2025-10-07-task-systolic-landscape-scans.md).
- Requires certificate integration per [Rigorous certificates and conjecture work](2025-10-07-task-systolic-certificates.md).
- Consumes insights from [Machine learning program](2025-10-07-task-systolic-ml-program.md) for prioritising search regions.

## Status Tracking

Use this brief to record progress across the sub-directions; update status when the bundle of activities matures or spins into independent briefs.

