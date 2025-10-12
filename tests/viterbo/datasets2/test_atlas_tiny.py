from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from viterbo.datasets2 import atlas_tiny
from viterbo.datasets2 import generators, quantities


def test_build_returns_expected_structure() -> None:
    dataset = atlas_tiny.build()
    assert dataset.num_rows == len(atlas_tiny.rows())
    assert dataset.num_rows == 8
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
        "volume[reference]",
        "volume[fast]",
    }
    assert set(dataset.column_names) == expected_columns

    first = dataset[0]
    assert first["vertices"], "vertices column should contain data"
    assert first["normals"], "normals column should contain data"
    assert first["offsets"], "offsets column should contain data"
    assert first["incidence"], "incidence column should contain data"


def test_rows_roundtrip_to_jax() -> None:
    row = atlas_tiny.rows()[0]
    vertices = jnp.asarray(row["vertices"], dtype=jnp.float64)
    normals = jnp.asarray(row["normals"], dtype=jnp.float64)
    offsets = jnp.asarray(row["offsets"], dtype=jnp.float64)
    incidence = jnp.asarray(row["incidence"], dtype=bool)

    assert vertices.ndim == 2
    assert normals.ndim == 2
    assert offsets.ndim == 1
    assert incidence.ndim == 2


def test_generators_return_polytope_sample() -> None:
    sample = generators.hypercube(dimension=2, radius=1.0)
    vertices, normals, offsets, incidence = sample.as_tuple()

    assert sample.vertices is vertices
    assert vertices.shape[1] == 2
    assert normals.shape[0] == offsets.shape[0]
    assert incidence.shape == (normals.shape[0], vertices.shape[0])

    vol_vertices = quantities.volume_from_vertices(vertices)
    vol_halfspaces = quantities.volume_from_halfspaces(normals, offsets, method="reference")
    assert vol_vertices == pytest.approx(vol_halfspaces)


def test_random_generators_cover_all_cases() -> None:
    halfspace = generators.sample_halfspace(
        jax.random.PRNGKey(0), dimension=2, num_facets=6, num_samples=2
    )
    assert len(halfspace) == 2
    assert all(sample.vertices.size > 0 for sample in halfspace)

    tangent = generators.sample_halfspace_tangent(
        jax.random.PRNGKey(1), dimension=2, num_facets=5, num_samples=1
    )
    assert len(tangent) == 1
    assert all(sample.vertices.size > 0 for sample in tangent)

    sphere = generators.sample_uniform_sphere(
        jax.random.PRNGKey(2), dimension=2, num_samples=1
    )
    assert len(sphere) == 1
    assert all(sample.vertices.size > 0 for sample in sphere)

    ball = generators.sample_uniform_ball(
        jax.random.PRNGKey(3), dimension=2, num_samples=1
    )
    assert len(ball) == 1
    assert all(sample.vertices.size > 0 for sample in ball)

    product = generators.enumerate_product_ngons(3, 3, 1)
    assert product
    dims = {sample.vertices.shape[1] for sample in product}
    assert dims == {4}
