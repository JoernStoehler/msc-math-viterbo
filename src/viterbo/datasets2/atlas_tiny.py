"""Atlas rows exposing every currently implemented algorithm."""

from __future__ import annotations

import json
import math

import jax
from datasets import Dataset, Features, Sequence, Value

from viterbo.datasets2 import converters, generators, quantities

_BUILDER_VERSION = "atlas_tiny@v2"
_SPECTRUM_HEAD = 8
_CYCLE_HEAD = 16

_BASE_FEATURES: dict[str, object] = {
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
    "tags": {
        "family": Value("string"),
        "dimension": Value("int32"),
        "normalized": Value("bool"),
        "symmetry_class": Value("string"),
    },
    "provenance": {
        "builder": Value("string"),
        "generator": Value("string"),
        "generator_config": Value("string"),
        "seed": Value("int64"),
    },
}

for algorithm in quantities.VOLUME_ALGORITHMS:
    _BASE_FEATURES[f"volume__{algorithm}"] = Value("float64")

for algorithm in quantities.CAPACITY_EHZ_ALGORITHMS:
    _BASE_FEATURES[f"capacity_ehz__{algorithm}"] = Value("float64")

for algorithm in quantities.SPECTRUM_ALGORITHMS:
    _BASE_FEATURES[f"spectrum__{algorithm}"] = Sequence(Value("float64"))

for algorithm in quantities.REEB_CYCLE_ALGORITHMS:
    _BASE_FEATURES[f"reeb_cycles__{algorithm}"] = {
        "edge_count": Value("int32"),
        "cycles": Sequence(Sequence(Value("int32"))),
    }

for algorithm in quantities.SYSTOLIC_RATIO_ALGORITHMS:
    _BASE_FEATURES[f"systolic_ratio__{algorithm}"] = Value("float64")

_DATASET_FEATURES = Features(_BASE_FEATURES)


def _row_from_sample(
    identifier: str,
    generator_name: str,
    config: dict[str, object],
    sample: generators.PolytopeSample,
) -> dict[str, object]:
    jax_row = quantities.AtlasJaxRow(
        vertices=sample.vertices,
        normals=sample.normals,
        offsets=sample.offsets,
        incidence=sample.incidence,
        volume={},
        capacity_ehz={},
    )

    volume_values: dict[str, float] = {}
    for algorithm in quantities.VOLUME_ALGORITHMS:
        volume_values[algorithm] = _safe_volume(jax_row, algorithm)
    jax_row.volume = volume_values

    capacity_values: dict[str, float] = {}
    for algorithm in quantities.CAPACITY_EHZ_ALGORITHMS:
        capacity_values[algorithm] = _safe_capacity(jax_row, algorithm)
    jax_row.capacity_ehz = capacity_values

    spectrum_values: dict[str, list[float]] = {}
    for algorithm in quantities.SPECTRUM_ALGORITHMS:
        values = _safe_spectrum(jax_row, algorithm, head=_SPECTRUM_HEAD)
        spectrum_values[algorithm] = [float(v) for v in values]

    cycle_values: dict[str, quantities.ReebCycleSummary] = {}
    for algorithm in quantities.REEB_CYCLE_ALGORITHMS:
        cycle_values[algorithm] = _safe_cycles(jax_row, algorithm, limit=_CYCLE_HEAD)

    systolic_values: dict[str, float] = {}
    for algorithm in quantities.SYSTOLIC_RATIO_ALGORITHMS:
        systolic_values[algorithm] = _safe_systolic(jax_row, algorithm)

    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    base = {
        "polytope_id": identifier,
        "generator": generator_name,
        "generator_config": config_json,
        "dimension": int(sample.vertices.shape[1]),
        "num_vertices": int(sample.vertices.shape[0]),
        "num_facets": int(sample.normals.shape[0]),
        "vertices": converters.array_to_python(sample.vertices),
        "normals": converters.array_to_python(sample.normals),
        "offsets": converters.array_to_python(sample.offsets),
        "incidence": converters.bool_array_to_python(sample.incidence),
        "tags": _tags_from_sample(generator_name, sample, config),
        "provenance": _provenance_record(generator_name, config_json, config),
    }

    for algorithm, value in volume_values.items():
        base[f"volume__{algorithm}"] = float(value)

    for algorithm, value in capacity_values.items():
        base[f"capacity_ehz__{algorithm}"] = float(value)

    for algorithm, values in spectrum_values.items():
        base[f"spectrum__{algorithm}"] = values

    for algorithm, summary in cycle_values.items():
        base[f"reeb_cycles__{algorithm}"] = {
            "edge_count": int(summary.edge_count),
            "cycles": [list(map(int, cycle)) for cycle in summary.cycles],
        }

    for algorithm, value in systolic_values.items():
        base[f"systolic_ratio__{algorithm}"] = float(value)

    return base


def _safe_volume(row: quantities.AtlasJaxRow, algorithm: str) -> float:
    try:
        return quantities.compute_volume(row, algorithm=algorithm)
    except (KeyError, ValueError):
        return float("nan")


def _safe_capacity(row: quantities.AtlasJaxRow, algorithm: str) -> float:
    try:
        return quantities.compute_capacity_ehz(row, algorithm=algorithm)
    except (KeyError, ValueError):
        return float("nan")


def _safe_spectrum(
    row: quantities.AtlasJaxRow, algorithm: str, *, head: int
) -> tuple[float, ...]:
    try:
        return quantities.compute_spectrum(row, algorithm=algorithm, head=head)
    except (KeyError, ValueError):
        return tuple()


def _safe_cycles(
    row: quantities.AtlasJaxRow, algorithm: str, *, limit: int
) -> quantities.ReebCycleSummary:
    try:
        return quantities.compute_reeb_cycles(row, algorithm=algorithm, limit=limit)
    except (KeyError, ValueError):
        return quantities.ReebCycleSummary(edge_count=0, cycles=tuple())


def _safe_systolic(row: quantities.AtlasJaxRow, algorithm: str) -> float:
    if row.capacity_ehz is None or algorithm not in row.capacity_ehz:
        return float("nan")
    try:
        return quantities.compute_systolic_ratio(row, algorithm=algorithm)
    except (KeyError, ValueError):
        return float("nan")


def _tags_from_sample(
    generator_name: str,
    sample: generators.PolytopeSample,
    config: dict[str, object],
) -> dict[str, object]:
    dimension = int(sample.vertices.shape[1])
    normalized = _is_normalized(generator_name, config)
    symmetry_class = _symmetry_class(generator_name)
    return {
        "family": generator_name,
        "dimension": dimension,
        "normalized": normalized,
        "symmetry_class": symmetry_class,
    }


def _is_normalized(generator_name: str, config: dict[str, object]) -> bool:
    radius = config.get("radius")
    if isinstance(radius, (int, float)):
        return math.isclose(float(radius), 1.0, rel_tol=0.0, abs_tol=1e-9)
    if generator_name in {"sample_uniform_sphere", "sample_uniform_ball"}:
        return True
    if generator_name.startswith("sample_"):
        return False
    return True


def _symmetry_class(generator_name: str) -> str:
    mapping = {
        "hypercube": "zonotope",
        "cross_polytope": "simplicial",
        "simplex": "simplicial",
        "enumerate_product_ngons": "product",
    }
    return mapping.get(generator_name, generator_name)


def _provenance_record(
    generator_name: str,
    config_json: str,
    config: dict[str, object],
) -> dict[str, object]:
    seed_value = config.get("seed", -1)
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        seed = -1
    return {
        "builder": _BUILDER_VERSION,
        "generator": generator_name,
        "generator_config": config_json,
        "seed": seed,
    }


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
