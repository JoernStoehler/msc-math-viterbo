"""Performance regression tests for Minkowski billiard solvers."""

from __future__ import annotations

import math

import pytest
import pytest_benchmark.plugin  # type: ignore[reportMissingTypeStubs]

from viterbo.geometry.polytopes import (
    Polytope,
    cartesian_product,
    cross_polytope,
    hypercube,
    regular_polygon_product,
)
from viterbo.symplectic.capacity.minkowski_billiards import (
    compute_minkowski_billiard_length_fast,
    compute_minkowski_billiard_length_reference,
)

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
    reference = compute_minkowski_billiard_length_reference(table, geometry)
    fast_result = benchmark(lambda: compute_minkowski_billiard_length_fast(table, geometry))
    assert math.isclose(fast_result, reference, rel_tol=1e-9, abs_tol=1e-9)
