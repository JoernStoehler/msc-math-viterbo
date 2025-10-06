"""Regression tests for (K, T)-Minkowski billiard solvers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from viterbo.geometry.polytopes import (
    cartesian_product,
    cross_polytope,
    hypercube,
    regular_polygon_product,
)
from viterbo.symplectic.capacity.minkowski_billiards import (
    build_normal_fan,
    compute_minkowski_billiard_length_fast,
    compute_minkowski_billiard_length_reference,
)
from viterbo.symplectic.capacity.minkowski_billiards.reference import (
    _canonical_cycle,
    _enumerate_cycles,
)


@pytest.mark.smoke
@pytest.mark.parametrize("dimension", [2, 3])
def test_normal_fan_vertex_adjacency_for_hypercube(dimension: int) -> None:
    """Hypercubes have one normal cone per vertex with expected degree."""

    cube = hypercube(dimension)
    fan = build_normal_fan(cube)
    assert fan.vertex_count == 2**dimension

    expected_degree = dimension
    for neighbors in fan.neighbors:
        assert len(neighbors) == expected_degree


@pytest.mark.smoke
def test_reference_minkowski_length_matches_hanner_bound() -> None:
    """Hypercube × cross-polytope trajectories minimise at four bounces."""

    table = hypercube(2)
    geometry = cross_polytope(2)

    length_reference = compute_minkowski_billiard_length_reference(table, geometry)
    length_fast = compute_minkowski_billiard_length_fast(table, geometry)

    assert math.isclose(length_reference, 8.0, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(length_fast, length_reference, rel_tol=1e-10, abs_tol=1e-10)

    milp_upper = 8.0
    assert length_fast <= milp_upper + 1e-10


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sides",
    [
        (4, 4),
        (4, 6),
    ],
    ids=["squarexsquare", "squarexhexagon"],
)
def test_fast_solver_matches_reference_on_polygon_products(sides: tuple[int, int]) -> None:
    """Reference enumeration and fast solver agree on polygon products."""

    table = regular_polygon_product(*sides)
    geometry = regular_polygon_product(*sides)

    length_reference = compute_minkowski_billiard_length_reference(table, geometry)
    length_fast = compute_minkowski_billiard_length_fast(table, geometry)

    assert np.isclose(length_fast, length_reference, rtol=1e-9, atol=1e-9)


@pytest.mark.smoke
def test_product_decomposition_matches_block_sum() -> None:
    """Product detection matches the block trajectory length per factor."""

    table_block = hypercube(2)
    geometry_block = cross_polytope(2)

    table_product = cartesian_product(table_block, table_block)
    geometry_product = cartesian_product(geometry_block, geometry_block)

    block_length = compute_minkowski_billiard_length_fast(table_block, geometry_block)
    length_fast = compute_minkowski_billiard_length_fast(table_product, geometry_product)
    length_reference = compute_minkowski_billiard_length_reference(table_product, geometry_product)

    assert math.isclose(block_length, 8.0, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(length_fast, block_length, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(length_reference, length_fast, rel_tol=1e-9, abs_tol=1e-9)


def test_cycle_enumeration_avoids_duplicate_orientations() -> None:
    """Johnson-style cycle enumeration yields unique canonical cycles."""

    fan = build_normal_fan(hypercube(2))
    cycles = list(_enumerate_cycles(fan, max_length=4))
    canonical_cycles = {_canonical_cycle(tuple(cycle)) for cycle in cycles}

    assert len(cycles) == len(canonical_cycles)


def test_cycle_enumeration_excludes_two_bounce_paths() -> None:
    """Cycles shorter than three bounces are pruned from enumeration."""

    fan = build_normal_fan(hypercube(3))
    cycles = list(_enumerate_cycles(fan, max_length=4))

    assert all(len(cycle) >= 3 for cycle in cycles)
