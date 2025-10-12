"""Experiment utilities and reference training loops."""

from viterbo.experiments.logreg_toy import (
    LogisticRegressionConfig,
    LogisticRegressionWeights,
    TrainingTrace,
    evaluate,
    train_logistic_regression,
)

# Make re-exports count as "used" for strict type-checkers without relying on __all__ (banned).
_exported = (
    LogisticRegressionConfig,
    LogisticRegressionWeights,
    TrainingTrace,
    evaluate,
    train_logistic_regression,
)
