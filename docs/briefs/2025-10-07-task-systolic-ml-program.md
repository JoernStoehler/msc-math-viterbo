---
status: proposed
created: 2025-10-07
source: chatgpt-pro-2025-10-07
summary: Develop surrogate models and hypothesis mining tools for systolic ratios.
---

# Machine learning surrogate program

## Motivation

Per the ChatGPT Pro plan, ML should accelerate exploration by predicting \(\text{sys}(X)\), flagging likely violations, and surfacing structural hypotheses.

## Scope

- Engineer polytope and Reeb-derived feature sets suitable for classical models and graph neural networks.
- Train calibrated predictors (e.g., gradient boosting, conformal wrappers, GNNs) on datasets from [Landscape scans](2025-10-07-task-systolic-landscape-scans.md).
- Explore hypothesis mining tools (mutual information estimators, symbolic regression) to propose interpretable relations.

## Deliverables

1. Feature extraction library referencing inputs from the systolic pipeline.
2. Model training scripts with evaluation dashboards and uncertainty calibration metrics.
3. Hypothesis reports linking mined relations to conjecture work in [Rigorous certificates](2025-10-07-task-systolic-certificates.md).

## Dependencies & Links

- Requires data from [Landscape scans and reproductions](2025-10-07-task-systolic-landscape-scans.md) and certificates from [Rigorous certificates and conjecture work](2025-10-07-task-systolic-certificates.md).
- Relies on implementations from [Systolic computation pipeline](2025-10-07-task-systolic-pipeline.md) for feature generation.
- Informs optimisation strategies in [Advanced optimization and analysis directions](2025-10-07-task-systolic-advanced-directions.md).

## Status Tracking

Mark progress as datasets become available, models train successfully, and hypothesis mining yields actionable leads.

