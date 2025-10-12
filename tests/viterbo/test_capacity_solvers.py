"""Regression tests for modern capacity solver wrappers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from viterbo import capacity, polytopes
from viterbo.types import Polytope


EXPECTED_SIMPLEX_CAPACITY = 1.0
EXPECTED_MINKOWSKI_LENGTH = 8.0


def _square_bundle(edge: float = 1.0) -> Polytope:
    vertices = jnp.asarray(
        [
            [edge, edge],
            [edge, -edge],
            [-edge, edge],
            [-edge, -edge],
        ],
        dtype=jnp.float64,
    )
    return polytopes.build_from_vertices(vertices)


def _diamond_bundle(edge: float = 1.0) -> Polytope:
    vertices = jnp.asarray(
        [
            [edge, 0.0],
            [-edge, 0.0],
            [0.0, edge],
            [0.0, -edge],
        ],
        dtype=jnp.float64,
    )
    return polytopes.build_from_vertices(vertices)


def _simplex_4d(edge: float = 2.0) -> Polytope:
    vertices = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [edge, 0.0, 0.0, 0.0],
            [0.0, edge, 0.0, 0.0],
            [0.0, 0.0, edge, 0.0],
            [0.0, 0.0, 0.0, edge],
        ],
        dtype=jnp.float64,
    )
    return polytopes.build_from_vertices(vertices)


def _unit_disk_vertices(samples: int = 64) -> jnp.ndarray:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, samples, endpoint=False)
    return jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1).astype(jnp.float64)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_facet_normal_solvers_agree_on_simplex() -> None:
    """Facet-normal reference and fast solvers agree on a 4D simplex."""
    bundle = _simplex_4d(edge=2.0)
    B_matrix = bundle.normals
    offsets = bundle.offsets
    reference = capacity.ehz_capacity_reference_facet_normals(B_matrix, offsets)
    fast = capacity.ehz_capacity_fast_facet_normals(B_matrix, offsets)
    assert fast == pytest.approx(reference, rel=1e-12, abs=0.0)
    assert reference == pytest.approx(EXPECTED_SIMPLEX_CAPACITY, rel=1e-12, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_milp_fast_matches_reference_upper_bound() -> None:
    """MILP fast solver returns same upper bound as exhaustive enumeration."""
    bundle = _simplex_4d(edge=2.0)
    B_matrix = bundle.normals
    offsets = bundle.offsets
    reference = capacity.ehz_capacity_reference_milp(B_matrix, offsets)
    fast = capacity.ehz_capacity_fast_milp(B_matrix, offsets, node_limit=128)
    assert fast[1] == pytest.approx(reference[1], rel=1e-12, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_reeb_cycle_fast_matches_reference() -> None:
    """Reeb-cycle wrappers match the facet-normal reference on a 4D simplex."""
    bundle = _simplex_4d(edge=2.0)
    B_matrix = bundle.normals
    offsets = bundle.offsets
    reference = capacity.ehz_capacity_reference_reeb(B_matrix, offsets)
    fast = capacity.ehz_capacity_fast_reeb(B_matrix, offsets)
    assert fast == pytest.approx(reference, rel=1e-12, abs=0.0)
    assert reference == pytest.approx(EXPECTED_SIMPLEX_CAPACITY, rel=1e-12, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_support_relaxation_variants_nonnegative() -> None:
    """Support-relaxation solvers return non-negative upper bounds near π."""
    vertices = _unit_disk_vertices(samples=48)
    bundle = polytopes.build_from_vertices(vertices)
    fast = capacity.support_relaxation_capacity_fast(
        bundle.normals,
        bundle.offsets,
        bundle.vertices,
        initial_density=7,
        refinement_steps=1,
        smoothing_parameters=(0.6, 0.3, 0.0),
        jit_compile=False,
    )
    reference = capacity.support_relaxation_capacity_reference(
        bundle.normals,
        bundle.offsets,
        bundle.vertices,
        grid_density=5,
        smoothing_parameters=(0.6, 0.3, 0.0),
        tolerance_sequence=(1e-3,),
        solver="SCS",
        center_vertices=True,
    )
    assert fast[0] >= 0.0
    assert reference[0] >= 0.0
    assert fast[0] == pytest.approx(jnp.pi, rel=0.2)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_symmetry_reduced_matches_reference_on_square() -> None:
    """Symmetry-reduced solvers agree with facet-normal reference on 4D simplex."""
    bundle = _simplex_4d(edge=2.0)
    B_matrix = bundle.normals
    offsets = bundle.offsets
    pairing = capacity.detect_opposite_facet_pairs(B_matrix)
    reference = capacity.ehz_capacity_reference_symmetry_reduced(B_matrix, offsets, pairing=pairing)
    fast = capacity.ehz_capacity_fast_symmetry_reduced(B_matrix, offsets, pairing=pairing)
    baseline = capacity.ehz_capacity_reference_facet_normals(B_matrix, offsets)
    assert reference == pytest.approx(baseline, rel=1e-12, abs=0.0)
    assert fast == pytest.approx(baseline, rel=1e-12, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_minkowski_billiard_lengths_match_reference() -> None:
    """Minkowski billiard fast solver matches the reference on square/diamond."""
    table = _square_bundle(edge=1.0)
    geometry = _diamond_bundle(edge=1.0)
    reference = capacity.minkowski_billiard_length_reference(table, geometry)
    fast = capacity.minkowski_billiard_length_fast(table, geometry)
    assert fast == pytest.approx(reference, rel=1e-12, abs=0.0)
    assert reference == pytest.approx(EXPECTED_MINKOWSKI_LENGTH, rel=1e-12, abs=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_available_solvers_list_contains_all_expected_keys() -> None:
    """available_solvers lists every solver identifier supported by adapters."""
    keys = set(capacity.available_solvers())
    expected = {
        "facet-normal-reference",
        "facet-normal-fast",
        "milp-reference",
        "milp-fast",
        "reeb-reference",
        "reeb-fast",
        "support-reference",
        "support-fast",
        "symmetry-reference",
        "symmetry-fast",
        "minkowski-reference",
        "minkowski-fast",
    }
    assert expected.issubset(keys)
