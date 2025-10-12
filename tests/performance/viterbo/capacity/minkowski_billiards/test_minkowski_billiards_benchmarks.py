"""Performance regression tests for Minkowski billiard solvers."""

from __future__ import annotations

import math

import pytest
import pytest_benchmark.plugin  # type: ignore[reportMissingTypeStubs]
import jax.numpy as jnp

from viterbo.geom import (
    Polytope,
    cartesian_product,
    cross_polytope,
    hypercube,
    regular_polygon_product,
)
from viterbo.capacity import minkowski_billiard_length_fast, minkowski_billiard_length_reference
from viterbo.types import Polytope as ModernPolytope

pytestmark = [pytest.mark.smoke, pytest.mark.deep]

_TABLE_GEOMETRY_PAIRS = [
    (
        "hypercube-cross-polytope-product",
        cartesian_product(hypercube(2), hypercube(2)),
        cartesian_product(cross_polytope(2), cross_polytope(2)),
    ),
    (
        "polygon-product-rot",
        cartesian_product(
            regular_polygon_product(4, 6),
            regular_polygon_product(4, 4),
        ),
        cartesian_product(
            regular_polygon_product(4, 6),
            regular_polygon_product(4, 4),
        ),
    ),
]
_TABLE_GEOMETRY_IDS = [entry[0] for entry in _TABLE_GEOMETRY_PAIRS]


@pytest.mark.goal_performance
@pytest.mark.smoke
@pytest.mark.benchmark(group="minkowski_billiards")
@pytest.mark.parametrize("label, table, geometry", _TABLE_GEOMETRY_PAIRS, ids=_TABLE_GEOMETRY_IDS)
def test_fast_solver_matches_reference_with_benchmark(
    benchmark: pytest_benchmark.plugin.BenchmarkFixture,
    label: str,
    table: Polytope,
    geometry: Polytope,
) -> None:
    """Benchmark the fast solver while validating against the reference enumeration."""

    assert isinstance(label, str)
    bundle_table = _to_bundle(table)
    bundle_geometry = _to_bundle(geometry)
    reference = minkowski_billiard_length_reference(bundle_table, bundle_geometry)
    fast_result = benchmark(lambda: minkowski_billiard_length_fast(bundle_table, bundle_geometry))
    assert math.isclose(fast_result, reference, rel_tol=1e-9, abs_tol=1e-9)


def _to_bundle(polytope: Polytope) -> ModernPolytope:
    B, c = polytope.halfspace_data()
    normals = jnp.asarray(B, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)
    dimension = normals.shape[1]
    return ModernPolytope(
        normals=normals,
        offsets=offsets,
        vertices=jnp.empty((0, dimension), dtype=jnp.float64),
        incidence=jnp.empty((0, normals.shape[0]), dtype=bool),
    )
