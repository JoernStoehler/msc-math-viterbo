"""Generate a small binary classification dataset for logistic regression experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import jax
from jax import config as jax_config
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SEED = 20251005

jax_config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class GaussianBlob:
    mean: tuple[float, float]
    cov_diag: tuple[float, float]


def _make_split(
    key: jax.Array,
    blob: GaussianBlob,
    size: int,
) -> jnp.ndarray:
    cov = jnp.diag(jnp.asarray(blob.cov_diag, dtype=jnp.float64))
    mean = jnp.asarray(blob.mean, dtype=jnp.float64)
    samples = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(size,))
    return samples


def _write_parquet(path: Path, features: jnp.ndarray, labels: jnp.ndarray) -> None:
    table = pa.Table.from_pydict(
        {
            "x0": pa.array(np.asarray(features[:, 0])),
            "x1": pa.array(np.asarray(features[:, 1])),
            "label": pa.array(np.asarray(labels)),
        }
    )
    pq.write_table(table, path)


@dataclass(frozen=True)
class DatasetConfig:
    version: str = "v1"
    train_size: int = 800
    test_size: int = 200
    positive_blob: GaussianBlob = GaussianBlob(mean=(1.5, 1.5), cov_diag=(0.5, 0.5))
    negative_blob: GaussianBlob = GaussianBlob(mean=(-1.0, -1.0), cov_diag=(0.75, 0.75))

    def as_serialisable(self) -> dict[str, object]:
        data = asdict(self)
        data["positive_blob"] = asdict(self.positive_blob)
        data["negative_blob"] = asdict(self.negative_blob)
        return data


def generate_dataset(output_dir: Path, seed: int, config: DatasetConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    key_train_pos, key_train_neg, key_test_pos, key_test_neg, key_train_perm, key_test_perm = (
        jax.random.split(key, 6)
    )

    pos_train = _make_split(key_train_pos, config.positive_blob, config.train_size // 2)
    neg_train = _make_split(key_train_neg, config.negative_blob, config.train_size // 2)
    train_features = jnp.concatenate([pos_train, neg_train], axis=0)
    train_labels = jnp.concatenate(
        [
            jnp.ones(pos_train.shape[0], dtype=jnp.int32),
            jnp.zeros(neg_train.shape[0], dtype=jnp.int32),
        ]
    )

    pos_test = _make_split(key_test_pos, config.positive_blob, config.test_size // 2)
    neg_test = _make_split(key_test_neg, config.negative_blob, config.test_size // 2)
    test_features = jnp.concatenate([pos_test, neg_test], axis=0)
    test_labels = jnp.concatenate(
        [
            jnp.ones(pos_test.shape[0], dtype=jnp.int32),
            jnp.zeros(neg_test.shape[0], dtype=jnp.int32),
        ]
    )

    # Shuffle within each split for randomness.
    train_perm = jax.random.permutation(key_train_perm, train_features.shape[0])
    test_perm = jax.random.permutation(key_test_perm, test_features.shape[0])
    train_features = train_features[train_perm]
    train_labels = train_labels[train_perm]
    test_features = test_features[test_perm]
    test_labels = test_labels[test_perm]

    _write_parquet(output_dir / "train.parquet", train_features, train_labels)
    _write_parquet(output_dir / "test.parquet", test_features, test_labels)

    metadata = {
        "dataset_name": "logreg-toy",
        "version": config.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "generator": "scripts/generate_logreg_toy_dataset.py",
        "jax_seed": seed,
        "split_sizes": {"train": int(config.train_size), "test": int(config.test_size)},
        "feature_names": ["x0", "x1"],
        "label_name": "label",
        "generation_params": config.as_serialisable(),
        "config_hash": _config_hash(config, seed),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# logreg-toy dataset ({config.version})

Synthetic 2D binary classification dataset for logistic regression smoke tests.

- **Positive blob**: mean {config.positive_blob.mean}, diagonal covariance {config.positive_blob.cov_diag}.
- **Negative blob**: mean {config.negative_blob.mean}, diagonal covariance {config.negative_blob.cov_diag}.
- **Splits**: train={config.train_size} samples, test={config.test_size} samples.
- **Seed**: {seed} (JAX PRNG).

Files:
- `train.parquet`
- `test.parquet`
- `metadata.json`

Generation script: `scripts/generate_logreg_toy_dataset.py`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _git_commit_hash() -> str | None:
    git_dir = Path(__file__).resolve().parents[1]
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=git_dir,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        return None
    return None


def _config_hash(config: DatasetConfig, seed: int) -> str:
    payload = json.dumps(
        {"config": config.as_serialisable(), "seed": seed},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artefacts/datasets/logreg-toy/v1"),
        help="Directory where dataset files will be written.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for generation")
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.seed, DatasetConfig())


if __name__ == "__main__":
    main()
