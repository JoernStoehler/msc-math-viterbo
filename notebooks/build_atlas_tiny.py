# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %% [markdown]
# # Build the ``atlas_tiny`` dataset
#
# Thin imperative notebook for creating the ``atlas_tiny`` dataset and collecting
# lightweight benchmark numbers. Artifacts are written under
# ``artefacts/datasets/atlas_tiny``.

# %%
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import jax
from datasets import Dataset

from viterbo.datasets2 import atlas_tiny, generators, quantities

ARTIFACT_DIR = Path("artefacts/datasets/atlas_tiny")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Build and persist the dataset.
start = perf_counter()
dataset = atlas_tiny.build()
build_duration = perf_counter() - start

dataset_path = ARTIFACT_DIR / "hf-dataset"
dataset.save_to_disk(dataset_path.as_posix())

print(f"Dataset rows: {dataset.num_rows}")
print(f"Build time: {build_duration:.6f} s")

# %%
# Benchmark each generator and quantity evaluation separately to understand bottlenecks.
timing_rows: list[dict[str, object]] = []

GENERATOR_SPECS = (
    ("hypercube-2", "hypercube", lambda: generators.hypercube(dimension=2, radius=1.0)),
    (
        "cross-polytope-2",
        "cross_polytope",
        lambda: generators.cross_polytope(dimension=2, radius=1.0),
    ),
    ("simplex-2", "simplex", lambda: generators.simplex(dimension=2)),
    (
        "random-halfspace-2",
        "sample_halfspace",
        lambda: generators.sample_halfspace(
            jax.random.PRNGKey(0),
            dimension=2,
            num_facets=6,
            num_samples=1,
        )[0],
    ),
    (
        "tangent-halfspace-2",
        "sample_halfspace_tangent",
        lambda: generators.sample_halfspace_tangent(
            jax.random.PRNGKey(1),
            dimension=2,
            num_facets=5,
            num_samples=1,
        )[0],
    ),
    (
        "sphere-hull-2",
        "sample_uniform_sphere",
        lambda: generators.sample_uniform_sphere(
            jax.random.PRNGKey(2),
            dimension=2,
            num_samples=1,
        )[0],
    ),
    (
        "ball-hull-2",
        "sample_uniform_ball",
        lambda: generators.sample_uniform_ball(
            jax.random.PRNGKey(3),
            dimension=2,
            num_samples=1,
        )[0],
    ),
    (
        "product-ngon-3x3",
        "enumerate_product_ngons",
        lambda: generators.enumerate_product_ngons(3, 3, 1)[0],
    ),
)

for identifier, algorithm_name, builder in GENERATOR_SPECS:
    start = perf_counter()
    sample = builder()
    normals = sample.normals
    offsets = sample.offsets
    timing_rows.append(
        {
            "category": "generator",
            "name": identifier,
            "algorithm": algorithm_name,
            "seconds": perf_counter() - start,
        }
    )

    for method in ("reference", "fast"):
        start = perf_counter()
        quantities.volume_from_halfspaces(normals, offsets, method=method)
        timing_rows.append(
            {
                "category": "quantity",
                "name": identifier,
                "algorithm": f"volume[{method}]",
                "seconds": perf_counter() - start,
            }
        )

# Record HuggingFace dataset append cost separately.
rows = list(atlas_tiny.rows())
start = perf_counter()
Dataset.from_list(rows, features=atlas_tiny.features())
timing_rows.append(
    {
        "category": "infra",
        "name": "huggingface",
        "algorithm": "Dataset.from_list",
        "seconds": perf_counter() - start,
    }
)

# Persist timing data and present a simple table.
timing_path = ARTIFACT_DIR / "timings.json"
timing_path.write_text(json.dumps(timing_rows, indent=2))

print("\nBenchmark summary (seconds):")
for row in timing_rows:
    print(f"- [{row['category']}] {row['name']:<16} {row['algorithm']:<18} {row['seconds']:.6e}")

# %%
# Return the dataset so the notebook preview shows the schema.
dataset
