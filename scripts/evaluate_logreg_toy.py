"""Evaluate a saved toy logistic regression checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import pyarrow.parquet as pq

from viterbo.experiments.logreg_toy import (
    evaluate,
    LogisticRegressionConfig,
    LogisticRegressionWeights,
)

jax.config.update("jax_enable_x64", True)

DEFAULT_DATASET_ROOT = Path("artefacts/datasets/logreg-toy")


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    data = table.to_pydict()
    features = np.stack(
        [np.asarray(data["x0"], dtype=np.float64), np.asarray(data["x1"], dtype=np.float64)], axis=1
    )
    labels = np.asarray(data["label"], dtype=np.float64)
    return features, labels


def _load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x, train_y = _load_split(dataset_dir / "train.parquet")
    test_x, test_y = _load_split(dataset_dir / "test.parquet")
    return train_x, train_y, test_x, test_y


def _restore_params(checkpoint_path: Path) -> LogisticRegressionWeights:
    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(str(checkpoint_path))
    params = state["params"]
    return LogisticRegressionWeights(
        weights=np.asarray(params["weights"]), bias=np.asarray(params["bias"])
    )


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    dataset_version = summary["dataset_version"]
    dataset_dir = (
        Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_ROOT / dataset_version
    )
    train_x, train_y, test_x, test_y = _load_dataset(dataset_dir)

    params = _restore_params(Path(summary["checkpoint_path"]))

    config = LogisticRegressionConfig(**summary["hyperparameters"])

    train_metrics = evaluate(params, train_x, train_y, l2_penalty=config.l2_penalty)
    test_metrics = evaluate(params, test_x, test_y, l2_penalty=config.l2_penalty)

    report = {
        "run_slug": summary["run_slug"],
        "dataset_version": dataset_version,
        "train": train_metrics,
        "test": test_metrics,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
