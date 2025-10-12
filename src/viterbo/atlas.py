"""The global atlas dataset, with conversion helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from jaxtyping import Array, Float

from viterbo.types import Cycle, Polytope
from viterbo.polytopes import incidence_matrix


ATLAS_FEATURES = Features(
    {
        # basic metadata
        "polytope_id": Value("string"),
        "notes": Value("string"),
        "distribution_name": Value("string"),
        # polytope representation
        "dimension": Value("int64"),
        "num_facets": Value("int64"),
        "num_vertices": Value("int64"),
        "normals": Sequence(Sequence(Value("float64"))),
        "offsets": Sequence(Value("float64")),
        "vertices": Sequence(Sequence(Value("float64"))),
        # quantities
        "volume": Value("float64"),
        "ehz_capacity": Value("float64"),
        "systolic_ratio": Value("float64"),
        "minimum_action_cycle": Sequence(Sequence(Value("float64"))),
    }
)


def atlas_features() -> Features:
    """Return the canonical Hugging Face dataset features for atlas rows."""

    return ATLAS_FEATURES


def build_dataset(rows_iterable: Iterable[Mapping[str, object]]) -> Dataset:
    """Construct a :class:`datasets.Dataset` from an iterable of row mappings."""

    rows = [dict(row) for row in rows_iterable]
    if rows:
        return Dataset.from_list(rows, features=ATLAS_FEATURES)
    empty_payload: dict[str, list[object]] = {name: [] for name in ATLAS_FEATURES.keys()}
    return Dataset.from_dict(empty_payload, features=ATLAS_FEATURES)


def append_rows(dataset: Dataset, rows_iterable: Iterable[Mapping[str, object]]) -> Dataset:
    """Append ``rows_iterable`` to ``dataset`` and return the combined dataset."""

    new_rows = [dict(row) for row in rows_iterable]
    if not new_rows:
        return dataset
    new_dataset = build_dataset(new_rows)
    return concatenate_datasets([dataset, new_dataset])


def save_dataset(dataset: Dataset, path: str) -> None:
    """Persist ``dataset`` to ``path`` using Hugging Face Arrow format."""

    dataset.save_to_disk(path)


def load_dataset(path: str) -> Dataset:
    """Load an atlas dataset that was previously saved via :func:`save_dataset`."""

    return Dataset.load_from_disk(path)


def map_quantities(
    dataset: Dataset, fn: Callable[[Mapping[str, object]], Mapping[str, object]]
) -> Dataset:
    """Apply ``fn`` to each row to update derived atlas quantities."""

    def _mapper(example: Mapping[str, object]) -> Mapping[str, object]:
        merged = dict(example)
        merged.update(fn(example))
        return merged

    return dataset.map(_mapper, features=ATLAS_FEATURES)


# we use different schemas for different dimensions
# to allow fixed-size arrays, batching, etc.
@dataclass(slots=True)
class AtlasRow:
    """A single row of the atlas dataset, parsed into JAX types."""

    # polytope metadata
    polytope_id: str
    dimension: int
    notes: str
    distribution_name: str
    # polytope representation
    polytope: Polytope
    # quantities
    volume: Float[Array, ""]
    ehz_capacity: Float[Array, ""]
    systolic_ratio: Float[Array, ""]
    minimum_action_cycle: Cycle


def as_polytope(
    dimension: int,
    num_facets: int,
    num_vertices: int,
    normals: list[Float[Array, " dimension"]],
    offsets: list[float],
    vertices: list[Float[Array, " dimension"]],
) -> Polytope:
    """Convert row fields into a JAX-first `Polytope`.

    Ensures float64 dtype and validates shapes before computing the facet–
    vertex incidence via the half-space test.
    """
    _normals = jnp.asarray(normals, dtype=jnp.float64)
    _offsets = jnp.asarray(offsets, dtype=jnp.float64)
    _vertices = jnp.asarray(vertices, dtype=jnp.float64)
    assert _normals.shape == (num_facets, dimension)
    assert _offsets.shape == (num_facets,)
    assert _vertices.shape == (num_vertices, dimension)
    _incidence = incidence_matrix(_normals, _offsets, _vertices)
    return Polytope(
        normals=_normals,
        offsets=_offsets,
        vertices=_vertices,
        incidence=_incidence,
    )


def as_cycle(
    dimension: int,
    num_points: int,
    points: list[Float[Array, " dimension"]],
    polytope: Polytope,
) -> Cycle:
    """Convert row fields into a `Cycle` linked to `polytope`.

    Computes point–facet incidence with consistent tolerances used by the
    polytope helper.
    """
    _points = jnp.asarray(points, dtype=jnp.float64)
    assert _points.shape == (num_points, dimension)
    normals = polytope.normals
    offsets = polytope.offsets
    incidence = incidence_matrix(normals, offsets, _points)
    return Cycle(points=_points, incidence=incidence)
