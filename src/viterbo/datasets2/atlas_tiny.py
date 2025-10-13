"""A small atlas dataset covering a representative polytope from each generator."""

from __future__ import annotations

import json
import math

import jax
from datasets import Dataset, Features, Sequence, Value

from viterbo._wrapped import spatial as _spatial
from viterbo.datasets2 import converters, generators

_DATASET_FEATURES = Features(
    {
        "polytope_id": Value("string"),
        "generator": Value("string"),
        "generator_config": Value("string"),
        "dimension": Value("int32"),
        "num_vertices": Value("int32"),
        "num_facets": Value("int32"),
        "vertices": Sequence(Sequence(Value("float64"))),
        "normals": Sequence(Sequence(Value("float64"))),
        "offsets": Sequence(Value("float64")),
        "incidence": Sequence(Sequence(Value("bool"))),
        "volume[reference]": Value("float64"),
        "volume[fast]": Value("float64"),
    }
)


def _row_from_sample(
    identifier: str,
    generator_name: str,
    config: dict[str, object],
    sample: generators.PolytopeSample,
) -> dict[str, object]:
    volume_value = float(_spatial.convex_hull_volume(sample.vertices))
    dimension = int(sample.vertices.shape[1])
    row = {
        "polytope_id": identifier,
        "generator": generator_name,
        "generator_config": json.dumps(config, sort_keys=True),
        "dimension": dimension,
        "num_vertices": int(sample.vertices.shape[0]),
        "num_facets": int(sample.normals.shape[0]),
        "vertices": converters.array_to_python(sample.vertices),
        "normals": converters.array_to_python(sample.normals),
        "offsets": converters.array_to_python(sample.offsets),
        "incidence": converters.bool_array_to_python(sample.incidence),
        "volume[reference]": volume_value,
        "volume[fast]": volume_value,
    }
    return row


def build() -> Dataset:
    """Return the tiny atlas dataset."""

    return Dataset.from_list(list(rows()), features=_DATASET_FEATURES)


def rows() -> tuple[dict[str, object], ...]:
    """Return the raw rows used to create the dataset."""

    cases: list[tuple[str, str, dict[str, object], generators.PolytopeSample]] = []

    dimension = 4

    halfspace_sample = generators.sample_halfspace(
        jax.random.PRNGKey(0),
        dimension=dimension,
        num_facets=10,
        num_samples=1,
    )[0]
    cases.append(
        (
            "random-halfspace-4",
            "sample_halfspace",
            {
                "dimension": dimension,
                "num_facets": 10,
                "num_samples": 1,
                "seed": 0,
            },
            halfspace_sample,
        )
    )

    tangent_sample = generators.sample_halfspace_tangent(
        jax.random.PRNGKey(1),
        dimension=dimension,
        num_facets=8,
        num_samples=1,
    )[0]
    cases.append(
        (
            "tangent-halfspace-4",
            "sample_halfspace_tangent",
            {
                "dimension": dimension,
                "num_facets": 8,
                "num_samples": 1,
                "seed": 1,
            },
            tangent_sample,
        )
    )

    sphere_sample = generators.sample_uniform_sphere(
        jax.random.PRNGKey(2),
        dimension=dimension,
        num_samples=1,
    )[0]
    cases.append(
        (
            "sphere-hull-4",
            "sample_uniform_sphere",
            {"dimension": dimension, "num_samples": 1, "seed": 2},
            sphere_sample,
        )
    )

    ball_sample = generators.sample_uniform_ball(
        jax.random.PRNGKey(3),
        dimension=dimension,
        num_samples=1,
    )[0]
    cases.append(
        (
            "ball-hull-4",
            "sample_uniform_ball",
            {"dimension": dimension, "num_samples": 1, "seed": 3},
            ball_sample,
        )
    )

    max_ngon_P = 5
    max_ngon_Q = 5
    max_rotation_Q = 6
    product_samples = generators.enumerate_product_ngons(
        max_ngon_P=max_ngon_P,
        max_ngon_Q=max_ngon_Q,
        max_rotation_Q=max_rotation_Q,
    )
    enumeration_metadata: list[dict[str, object]] = []
    for k_P in range(3, max_ngon_P + 1):
        for k_Q in range(3, max_ngon_Q + 1):
            for s in range(1, max_rotation_Q + 1):
                for r in range(0, s):
                    if r == 0 and s != 1:
                        continue
                    if r != 0 and math.gcd(r, s) != 1:
                        continue
                    if r / s >= 1.0 / k_Q:
                        continue
                    angle = 2.0 * math.pi * r / s
                    enumeration_metadata.append(
                        {
                            "max_ngon_P": max_ngon_P,
                            "max_ngon_Q": max_ngon_Q,
                            "max_rotation_Q": max_rotation_Q,
                            "k_P": k_P,
                            "k_Q": k_Q,
                            "rotation": {
                                "numerator": r,
                                "denominator": s,
                            },
                            "rotation_radians": angle,
                        }
                    )

    target_index = next(
        (
            index
            for index, meta in enumerate(enumeration_metadata)
            if meta["k_P"] == 5
            and meta["k_Q"] == 5
            and meta["rotation"]["numerator"] == 1
            and meta["rotation"]["denominator"] == 6
        ),
        None,
    )
    if target_index is None:
        raise RuntimeError("Failed to locate requested product-ngon configuration")
    product_sample = product_samples[target_index]
    product_config = enumeration_metadata[target_index]
    cases.append(
        (
            "product-ngon-5x5-rot-1-6",
            "enumerate_product_ngons",
            product_config,
            product_sample,
        )
    )

    pentagon_sample = generators.pentagon_product_4d_sample(rotation=math.pi / 2.0)
    cases.append(
        (
            "pentagon-product-counterexample",
            "pentagon_product_4d",
            {
                "rotation_radians": math.pi / 2.0,
                "rotation_degrees": 90.0,
                "radius": 1.0,
            },
            pentagon_sample,
        )
    )

    return tuple(
        _row_from_sample(identifier, generator_name, config, sample)
        for identifier, generator_name, config, sample in cases
    )


def features() -> Features:
    """Return the dataset feature schema."""

    return _DATASET_FEATURES
