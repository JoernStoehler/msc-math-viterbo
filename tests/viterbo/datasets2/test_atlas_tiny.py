from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from viterbo.datasets2 import atlas_tiny, generators, quantities


@pytest.mark.goal_code
@pytest.mark.smoke
def test_build_returns_expected_structure() -> None:
    """Atlas tiny build returns rows with per-algorithm columns."""

    dataset = atlas_tiny.build()
    assert dataset.num_rows == len(atlas_tiny.rows()) == 8

    expected_columns = {
        "polytope_id",
        "generator",
        "generator_config",
        "dimension",
        "num_vertices",
        "num_facets",
        "vertices",
        "normals",
        "offsets",
        "incidence",
        "tags",
        "provenance",
    }
    expected_columns.update({f"volume__{a}" for a in quantities.VOLUME_ALGORITHMS})
    expected_columns.update({f"capacity_ehz__{a}" for a in quantities.CAPACITY_EHZ_ALGORITHMS})
    expected_columns.update({f"spectrum__{a}" for a in quantities.SPECTRUM_ALGORITHMS})
    expected_columns.update({f"reeb_cycles__{a}" for a in quantities.REEB_CYCLE_ALGORITHMS})
    expected_columns.update({f"systolic_ratio__{a}" for a in quantities.SYSTOLIC_RATIO_ALGORITHMS})
    assert set(dataset.column_names) == expected_columns

    first = dataset[0]
    assert first["vertices"], "vertices column should contain data"
    assert first["normals"], "normals column should contain data"
    assert first["offsets"], "offsets column should contain data"
    assert first["incidence"], "incidence column should contain data"

    for algorithm in quantities.VOLUME_ALGORITHMS:
        assert math.isnan(first[f"volume__{algorithm}"]) or first[f"volume__{algorithm}"] >= 0.0

    for algorithm in quantities.CAPACITY_EHZ_ALGORITHMS:
        assert algorithm in quantities.SYSTOLIC_RATIO_ALGORITHMS
        assert math.isnan(first[f"capacity_ehz__{algorithm}"]) or first[
            f"capacity_ehz__{algorithm}"
        ] >= 0.0

    for algorithm in quantities.SPECTRUM_ALGORITHMS:
        spectrum_values = first[f"spectrum__{algorithm}"]
        assert isinstance(spectrum_values, list)
        assert len(spectrum_values) <= 8

    for algorithm in quantities.REEB_CYCLE_ALGORITHMS:
        cycles_entry = first[f"reeb_cycles__{algorithm}"]
        assert set(cycles_entry.keys()) == {"edge_count", "cycles"}
        assert isinstance(cycles_entry["cycles"], list)

    assert set(first["tags"].keys()) == {
        "family",
        "dimension",
        "normalized",
        "symmetry_class",
    }
    assert set(first["provenance"].keys()) == {
        "builder",
        "generator",
        "generator_config",
        "seed",
    }


@pytest.mark.goal_code
@pytest.mark.smoke
def test_rows_roundtrip_to_jax() -> None:
    """Rows convert to JAX arrays and reproduce quantity dispatch."""

    row = atlas_tiny.rows()[0]
    vertices = jnp.asarray(row["vertices"], dtype=jnp.float64)
    normals = jnp.asarray(row["normals"], dtype=jnp.float64)
    offsets = jnp.asarray(row["offsets"], dtype=jnp.float64)
    incidence = jnp.asarray(row["incidence"], dtype=bool)

    assert vertices.ndim == 2
    assert normals.ndim == 2
    assert offsets.ndim == 1
    assert incidence.ndim == 2

    jax_row = quantities.AtlasJaxRow(
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        incidence=incidence,
        volume={},
        capacity_ehz={},
    )

    volume_expected: dict[str, float] = {}
    for algorithm in quantities.VOLUME_ALGORITHMS:
        volume_expected[algorithm] = quantities.compute_volume(jax_row, algorithm=algorithm)
    jax_row.volume = volume_expected

    capacity_expected: dict[str, float] = {}
    for algorithm in quantities.CAPACITY_EHZ_ALGORITHMS:
        try:
            capacity_expected[algorithm] = quantities.compute_capacity_ehz(
                jax_row, algorithm=algorithm
            )
        except ValueError:
            capacity_expected[algorithm] = float("nan")
    jax_row.capacity_ehz = capacity_expected

    for algorithm, expected in volume_expected.items():
        actual = row[f"volume__{algorithm}"]
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected)

    for algorithm, expected in capacity_expected.items():
        actual = row[f"capacity_ehz__{algorithm}"]
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected)

    for algorithm in quantities.SPECTRUM_ALGORITHMS:
        observed = row[f"spectrum__{algorithm}"]
        try:
            expected = quantities.compute_spectrum(
                jax_row, algorithm=algorithm, head=len(observed)
            )
        except ValueError:
            expected = tuple()
        assert observed == pytest.approx(expected)

    for algorithm in quantities.REEB_CYCLE_ALGORITHMS:
        observed = row[f"reeb_cycles__{algorithm}"]
        try:
            summary = quantities.compute_reeb_cycles(
                jax_row, algorithm=algorithm, limit=len(observed["cycles"]) + 8
            )
        except ValueError:
            summary = quantities.ReebCycleSummary(edge_count=0, cycles=tuple())
        assert observed["edge_count"] == summary.edge_count
        assert observed["cycles"] == [list(cycle) for cycle in summary.cycles]

    for algorithm in quantities.SYSTOLIC_RATIO_ALGORITHMS:
        observed = row[f"systolic_ratio__{algorithm}"]
        try:
            expected = quantities.compute_systolic_ratio(jax_row, algorithm=algorithm)
        except (KeyError, ValueError):
            expected = float("nan")
        if math.isnan(expected):
            assert math.isnan(observed)
        else:
            assert observed == pytest.approx(expected)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_generators_return_polytope_sample() -> None:
    """Generator helpers return PolytopeSample with consistent shapes and volume parity."""

    sample = generators.hypercube(dimension=2, radius=1.0)
    vertices, normals, offsets, incidence = sample.as_tuple()

    assert sample.vertices is vertices
    assert vertices.shape[1] == 2
    assert normals.shape[0] == offsets.shape[0]
    assert incidence.shape == (normals.shape[0], vertices.shape[0])

    vol_vertices = quantities.compute_volume(
        quantities.AtlasJaxRow(vertices=vertices), algorithm="vertices_reference"
    )
    vol_halfspaces = quantities.compute_volume(
        quantities.AtlasJaxRow(normals=normals, offsets=offsets),
        algorithm="halfspaces_reference",
    )
    assert vol_vertices == pytest.approx(vol_halfspaces)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_dispatch_requires_prerequisites() -> None:
    """Canonical dispatch raises when prerequisites are missing."""

    empty_row = quantities.AtlasJaxRow()
    with pytest.raises(KeyError):
        quantities.compute_volume(empty_row, algorithm="halfspaces_reference")
    with pytest.raises(KeyError):
        quantities.compute_capacity_ehz(empty_row, algorithm="facet_normals_reference")
    with pytest.raises(KeyError):
        quantities.compute_spectrum(empty_row, algorithm="ehz_reference", head=8)
    with pytest.raises(KeyError):
        quantities.compute_reeb_cycles(empty_row, algorithm="oriented_edges", limit=4)

    incomplete_row = quantities.AtlasJaxRow(volume={})
    incomplete_row.capacity_ehz = {"facet_normals_reference": 1.0}
    with pytest.raises(KeyError):
        quantities.compute_systolic_ratio(incomplete_row, algorithm="facet_normals_reference")


@pytest.mark.goal_code
@pytest.mark.smoke
def test_rows_benchmark_smoke(benchmark) -> None:
    """Profile atlas row construction to track future regressions."""

    result = benchmark(lambda: atlas_tiny.rows())
    assert len(result) == 8
