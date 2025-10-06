# pragma: no cover
"""JAX baseline logistic regression for toy datasets."""

# Toy module kept for documentation demos rather than regression tests.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Hyperparameters for full-batch logistic regression training."""

    learning_rate: float = 0.1
    num_steps: int = 500
    l2_penalty: float = 0.0


class TrainingTrace(NamedTuple):
    """Per-step metrics recorded during optimisation."""

    loss: Float[Array, " step"]
    accuracy: Float[Array, " step"]
    gradient_norm: Float[Array, " step"]


class LogisticRegressionWeights(NamedTuple):
    """Model parameters (weights and bias)."""

    weights: Float[Array, " features"]
    bias: Float[Array, ""]


def train_logistic_regression(
    features: Float[Array, " batch features"],
    labels: Float[Array, " batch"],
    config: LogisticRegressionConfig,
    *,
    key: jax.Array,
) -> tuple[LogisticRegressionWeights, TrainingTrace]:
    """Train logistic regression using gradient descent.

    Args:
      features: Training features array with shape (batch, features).
      labels: Binary labels (0 or 1) with shape (batch,).
      config: Hyperparameters controlling optimisation.
      key: PRNG key used to initialise parameters.

    Returns:
      Tuple of fitted parameters and a training trace for diagnostics.
    """

    features = jnp.asarray(features, dtype=jnp.float64)
    labels = jnp.asarray(labels, dtype=jnp.float64)

    num_features = features.shape[1]
    w_key, b_key = jax.random.split(key)
    params = LogisticRegressionWeights(
        weights=jax.random.normal(w_key, (num_features,), dtype=jnp.float64) * 0.01,
        bias=jax.random.normal(b_key, (), dtype=jnp.float64) * 0.01,
    )

    def loss_fn(params: LogisticRegressionWeights) -> tuple[jnp.ndarray, tuple[jnp.ndarray]]:
        logits = _predict_logits(params, features)
        probs = jnn.sigmoid(logits)
        eps = jnp.array(1e-8, dtype=jnp.float64)
        log_likelihood = -jnp.mean(
            labels * jnp.log(probs + eps) + (1.0 - labels) * jnp.log(1.0 - probs + eps)
        )
        l2 = config.l2_penalty * jnp.sum(jnp.square(params.weights))
        loss = log_likelihood + l2
        predictions = (probs >= 0.5).astype(jnp.float64)
        accuracy = jnp.mean(jnp.where(predictions == labels, 1.0, 0.0))
        return loss, (accuracy,)

    grad_loss = jax.value_and_grad(loss_fn, has_aux=True)

    losses: list[jnp.ndarray] = []
    accuracies: list[jnp.ndarray] = []
    grad_norms: list[jnp.ndarray] = []
    current = params
    for _ in range(config.num_steps):
        (loss_value, (accuracy_value,)), grads = grad_loss(current)
        new_weights = current.weights - config.learning_rate * grads.weights
        new_bias = current.bias - config.learning_rate * grads.bias
        current = LogisticRegressionWeights(new_weights, new_bias)
        grad_norm = jnp.linalg.norm(jnp.concatenate((grads.weights, grads.bias[None])))
        losses.append(loss_value)
        accuracies.append(accuracy_value)
        grad_norms.append(grad_norm)

    trace = TrainingTrace(
        loss=jnp.stack(losses),
        accuracy=jnp.stack(accuracies),
        gradient_norm=jnp.stack(grad_norms),
    )
    return current, trace


def _predict_logits(
    params: LogisticRegressionWeights,
    features: Float[Array, " batch features"],
) -> Float[Array, " batch"]:
    return features @ params.weights + params.bias


def evaluate(
    params: LogisticRegressionWeights,
    features: Float[Array, " batch features"],
    labels: Float[Array, " batch"],
    *,
    l2_penalty: float = 0.0,
) -> dict[str, float]:
    """Compute loss and accuracy on a holdout split."""

    features = jnp.asarray(features, dtype=jnp.float64)
    labels = jnp.asarray(labels, dtype=jnp.float64)
    logits = _predict_logits(params, features)
    probs = jnn.sigmoid(logits)
    eps = jnp.array(1e-8, dtype=jnp.float64)
    log_likelihood = -jnp.mean(
        labels * jnp.log(probs + eps) + (1.0 - labels) * jnp.log(1.0 - probs + eps)
    )
    l2 = l2_penalty * jnp.sum(jnp.square(params.weights))
    loss = log_likelihood + l2
    predictions = (probs >= 0.5).astype(jnp.float64)
    accuracy = jnp.mean(jnp.where(predictions == labels, 1.0, 0.0))
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
    }
