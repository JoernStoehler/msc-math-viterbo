"""Train the toy logistic regression baseline and log artefacts."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import pyarrow.parquet as pq
import wandb
from tensorboardX import SummaryWriter

from viterbo.experiments.logreg_toy import (
    LogisticRegressionConfig,
    evaluate,
    train_logistic_regression,
)

jax.config.update("jax_enable_x64", True)

PROJECT_NAME = "viterbo-experiments"
DEFAULT_DATASET_DIR = Path("artefacts/datasets/logreg-toy/v1")
OUTPUT_MODELS_DIR = Path("artefacts/models/logreg-toy")
METRICS_FILENAME = "metrics.jsonl"
SUMMARY_YAML_FILENAME = "summary.yaml"
SUMMARY_JSON_FILENAME = "summary.json"


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    data = table.to_pydict()
    features = np.stack(
        [np.asarray(data["x0"], dtype=np.float64), np.asarray(data["x1"], dtype=np.float64)], axis=1
    )
    labels = np.asarray(data["label"], dtype=np.float64)
    return features, labels


def _load_dataset(
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    train_features, train_labels = _load_split(base_dir / "train.parquet")
    test_features, test_labels = _load_split(base_dir / "test.parquet")
    return train_features, train_labels, test_features, test_labels, metadata


def _make_run_name(dataset_version: str, variant: str) -> str:
    date = datetime.now(UTC).strftime("%Y%m%d")
    return f"{date}_logreg_toy_{dataset_version}_{variant}"


def _make_run_slug(dataset_version: str, variant: str) -> str:
    date = datetime.now(UTC).strftime("%Y%m%d")
    token = secrets.token_hex(3)
    return f"{date}-logreg-toy-{dataset_version}-{variant}-{token}"


def _ensure_output_dir(slug: str) -> Path:
    run_dir = OUTPUT_MODELS_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _log_metrics_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry) + "\n")


def run(args: argparse.Namespace) -> None:
    """Train the toy logistic regression model and log artefacts."""
    dataset_dir = Path(args.dataset_dir)
    train_x, train_y, test_x, test_y, metadata = _load_dataset(dataset_dir)

    config = LogisticRegressionConfig(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        l2_penalty=args.l2_penalty,
    )

    run_name = _make_run_name(metadata["version"], args.variant)
    run_slug = _make_run_slug(metadata["version"], args.variant)
    run_dir = _ensure_output_dir(run_slug)

    wandb_run = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        tags=[
            "experiment:logreg_toy",
            f"dataset:{metadata['version']}",
            "model:logistic_regression",
            f"run_type:{args.variant}",
        ],
        config={
            "dataset": metadata,
            "training": asdict(config),
            "variant": args.variant,
        },
        dir=str(run_dir),
        reinit=True,
    )

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    key = jax.random.PRNGKey(args.init_seed)
    params, trace = train_logistic_regression(train_x, train_y, config, key=key)

    metrics_entries: list[dict[str, object]] = []
    for step in range(config.num_steps):
        metrics = {
            "train/loss": float(trace.loss[step]),
            "train/accuracy": float(trace.accuracy[step]),
            "train/grad_norm": float(trace.gradient_norm[step]),
        }
        wandb_run.log({**metrics, "step": step}, step=step)
        writer.add_scalar("train/loss", metrics["train/loss"], step)
        writer.add_scalar("train/accuracy", metrics["train/accuracy"], step)
        writer.add_scalar("train/grad_norm", metrics["train/grad_norm"], step)
        for metric_name, value in metrics.items():
            metrics_entries.append(
                {
                    "step": step,
                    "split": "train",
                    "metric": metric_name.split("/")[-1],
                    "value": value,
                }
            )

    eval_train = evaluate(params, train_x, train_y, l2_penalty=config.l2_penalty)
    eval_test = evaluate(params, test_x, test_y, l2_penalty=config.l2_penalty)

    wandb_run.log(
        {
            "train/final_loss": eval_train["loss"],
            "train/final_accuracy": eval_train["accuracy"],
            "test/loss": eval_test["loss"],
            "test/accuracy": eval_test["accuracy"],
        }
    )

    metrics_entries.append(
        {"step": config.num_steps, "split": "test", "metric": "loss", "value": eval_test["loss"]}
    )
    metrics_entries.append(
        {
            "step": config.num_steps,
            "split": "test",
            "metric": "accuracy",
            "value": eval_test["accuracy"],
        }
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step_{config.num_steps}"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        str(checkpoint_path.resolve()),
        {"params": {"weights": params.weights, "bias": params.bias}},
    )

    _log_metrics_jsonl(run_dir / METRICS_FILENAME, metrics_entries)

    summary = {
        "run_name": run_name,
        "run_slug": run_slug,
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_version": metadata["version"],
        "variant": args.variant,
        "hyperparameters": asdict(config),
        "train_metrics": eval_train,
        "test_metrics": eval_test,
        "checkpoint_step": config.num_steps,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "artefacts": {
            "metrics_jsonl": str((run_dir / METRICS_FILENAME).resolve()),
            "tensorboard": str((run_dir / "tensorboard").resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
        },
    }
    (run_dir / SUMMARY_YAML_FILENAME).write_text(_to_yaml(summary), encoding="utf-8")
    (run_dir / SUMMARY_JSON_FILENAME).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    wandb_run.log_artifact(
        _create_artifact(
            run_dir,
            run_slug,
            metadata,
            checkpoint_path=checkpoint_path,
        )
    )
    wandb_run.finish()
    writer.close()


def _create_artifact(
    run_dir: Path,
    run_slug: str,
    dataset_metadata: dict[str, object],
    *,
    checkpoint_path: Path,
) -> wandb.Artifact:
    artifact = wandb.Artifact(name=run_slug, type="model")
    artifact.add_dir(str(run_dir))
    artifact.metadata = {
        "dataset": dataset_metadata,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    return artifact


def _to_yaml(payload: dict[str, object]) -> str:
    try:
        import yaml  # type: ignore[import-not-found]

        return yaml.safe_dump(payload, sort_keys=False)
    except ImportError:
        # Minimal YAML writer to avoid new dependency if PyYAML is absent.
        lines = []
        for key, value in payload.items():
            lines.append(f"{key}: {json.dumps(value)}")
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for the training script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--l2-penalty", type=float, default=0.0)
    parser.add_argument("--variant", type=str, default="baseline")
    parser.add_argument("--init-seed", type=int, default=42)
    return parser


def main() -> None:
    """Parse arguments and run training."""
    parser = build_parser()
    args = parser.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    run(args)


if __name__ == "__main__":
    main()
