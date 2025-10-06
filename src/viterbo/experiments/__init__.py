"""Experiment utilities and reference training loops."""

from viterbo.experiments.logreg_toy import (  # noqa: F401 re-export
    LogisticRegressionConfig,
    LogisticRegressionWeights,
    TrainingTrace,
    evaluate,
    train_logistic_regression,
)

__all__ = [
    "LogisticRegressionConfig",
    "LogisticRegressionWeights",
    "TrainingTrace",
    "evaluate",
    "train_logistic_regression",
]
