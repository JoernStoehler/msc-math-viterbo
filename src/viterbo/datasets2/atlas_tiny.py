"""A small atlas dataset covering a representative polytope from each generator."""

from __future__ import annotations

import json

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

    cases.append(
        (
            "hypercube-2",
            "hypercube",
            {"dimension": 2, "radius": 1.0},
            generators.hypercube(dimension=2, radius=1.0),
        )
    )
    cases.append(
        (
            "cross-polytope-2",
            "cross_polytope",
            {"dimension": 2, "radius": 1.0},
            generators.cross_polytope(dimension=2, radius=1.0),
        )
    )
    cases.append(
        (
            "simplex-2",
            "simplex",
            {"dimension": 2},
            generators.simplex(dimension=2),
        )
    )

    halfspace_sample = generators.sample_halfspace(
        jax.random.PRNGKey(0),
        dimension=2,
        num_facets=6,
        num_samples=1,
    )[0]
    cases.append(
        (
            "random-halfspace-2",
            "sample_halfspace",
            {"dimension": 2, "num_facets": 6, "num_samples": 1, "seed": 0},
            halfspace_sample,
        )
    )

    tangent_sample = generators.sample_halfspace_tangent(
        jax.random.PRNGKey(1),
        dimension=2,
        num_facets=5,
        num_samples=1,
    )[0]
    cases.append(
        (
            "tangent-halfspace-2",
            "sample_halfspace_tangent",
            {"dimension": 2, "num_facets": 5, "num_samples": 1, "seed": 1},
            tangent_sample,
        )
    )

    sphere_sample = generators.sample_uniform_sphere(
        jax.random.PRNGKey(2),
        dimension=2,
        num_samples=1,
    )[0]
    cases.append(
        (
            "sphere-hull-2",
            "sample_uniform_sphere",
            {"dimension": 2, "num_samples": 1, "seed": 2},
            sphere_sample,
        )
    )

    ball_sample = generators.sample_uniform_ball(
        jax.random.PRNGKey(3),
        dimension=2,
        num_samples=1,
    )[0]
    cases.append(
        (
            "ball-hull-2",
            "sample_uniform_ball",
            {"dimension": 2, "num_samples": 1, "seed": 3},
            ball_sample,
        )
    )

    product_sample = generators.enumerate_product_ngons(
        max_ngon_P=3,
        max_ngon_Q=3,
        max_rotation_Q=1,
    )[0]
    cases.append(
        (
            "product-ngon-3x3",
            "enumerate_product_ngons",
            {"max_ngon_P": 3, "max_ngon_Q": 3, "max_rotation_Q": 1},
            product_sample,
        )
    )

    return tuple(
        _row_from_sample(identifier, generator_name, config, sample)
        for identifier, generator_name, config, sample in cases
    )


def features() -> Features:
    """Return the dataset feature schema."""

    return _DATASET_FEATURES

__all__ = ["build", "features", "rows"]
