"""Affine invariance and scaling tests for capacity, volume, and cycles."""

from __future__ import annotations
from itertools import combinations
from typing import Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from viterbo.cycles import minimum_cycle_reference
from viterbo.datasets import builders as polytopes
from viterbo.datasets.catalog import hypercube
from viterbo.math import symplectic, volume
from viterbo.math.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
)
from viterbo.math.capacity.milp import (
    ehz_capacity_fast_milp,
    ehz_capacity_reference_milp,
)
from viterbo.math.capacity.reeb_cycles import (
    ehz_capacity_fast_reeb,
    ehz_capacity_reference_reeb,
)
from viterbo.math.capacity.support_relaxation import (
    support_relaxation_capacity_fast,
    support_relaxation_capacity_reference,
)
from viterbo.math.capacity.symmetry_reduced import (
    ehz_capacity_fast_symmetry_reduced,
    ehz_capacity_reference_symmetry_reduced,
)

try:
    import scipy.spatial as spatial  # type: ignore[reportMissingTypeStubs]
except ModuleNotFoundError as exc:  # pragma: no cover - dependency is part of runtime image
    raise RuntimeError("scipy is required for convex hull construction in tests") from exc


CapacityEvaluator = Callable[[polytopes.Polytope], float]
VolumeEvaluator = Callable[[polytopes.Polytope], float]


def _random_convex_hull_vertices(seed: int, count: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 4))
    try:
        hull = spatial.ConvexHull(points)
    except spatial.QhullError:
        assume(False)
        raise AssertionError  # Unreachable due to assume
    hull_vertices = hull.vertices
    assume(hull_vertices.size >= 5)
    ordered = points[hull_vertices]
    return jnp.asarray(ordered, dtype=jnp.float64)


def _simplex_vertices(edge: float = 2.0) -> jnp.ndarray:
    return jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [edge, 0.0, 0.0, 0.0],
            [0.0, edge, 0.0, 0.0],
            [0.0, 0.0, edge, 0.0],
            [0.0, 0.0, 0.0, edge],
        ],
        dtype=jnp.float64,
    )


def _capacity_evaluators_for_invariance() -> Sequence[tuple[str, CapacityEvaluator]]:
    return (
        (
            "facet_reference",
            lambda P: float(ehz_capacity_reference_facet_normals(P.normals, P.offsets)),
        ),
        (
            "facet_fast",
            lambda P: float(ehz_capacity_fast_facet_normals(P.normals, P.offsets)),
        ),
        (
            "reeb_reference",
            lambda P: float(ehz_capacity_reference_reeb(P.normals, P.offsets)),
        ),
        (
            "reeb_fast",
            lambda P: float(ehz_capacity_fast_reeb(P.normals, P.offsets)),
        ),
        (
            "symmetry_reference",
            lambda P: float(ehz_capacity_reference_symmetry_reduced(P.normals, P.offsets)),
        ),
        (
            "symmetry_fast",
            lambda P: float(ehz_capacity_fast_symmetry_reduced(P.normals, P.offsets)),
        ),
        (
            "milp_reference",
            lambda P: float(ehz_capacity_reference_milp(P.normals, P.offsets)[1]),
        ),
        (
            "milp_fast",
            lambda P: float(ehz_capacity_fast_milp(P.normals, P.offsets)[1]),
        ),
        (
            "support_reference",
            lambda P: float(
                support_relaxation_capacity_reference(
                    P.normals, P.offsets, P.vertices
                )[0]
            ),
        ),
        (
            "support_fast",
            lambda P: float(
                support_relaxation_capacity_fast(P.normals, P.offsets, P.vertices)[0]
            ),
        ),
    )


def _capacity_evaluators_simplex() -> Sequence[tuple[str, CapacityEvaluator]]:
    evaluators = list(_capacity_evaluators_for_invariance())
    # Support-relaxation solvers provide loose upper bounds that collapse to zero on
    # simple polytopes; exclude them from exact matching checks.
    return tuple(
        (name, fn)
        for name, fn in evaluators
        if not name.startswith("support")
    )


def _volume_evaluators() -> Sequence[tuple[str, VolumeEvaluator]]:
    return (
        ("volume_reference", lambda P: float(volume.volume_reference(P.vertices))),
        (
            "polytope_volume_reference",
            lambda P: float(volume.polytope_volume_reference(P.normals, P.offsets)),
        ),
        (
            "polytope_volume_fast",
            lambda P: float(volume.polytope_volume_fast(P.normals, P.offsets)),
        ),
    )


def _sorted_points(points: jnp.ndarray) -> jnp.ndarray:
    if points.size == 0:
        return points
    indices = jnp.lexsort(tuple(points[:, idx] for idx in reversed(range(points.shape[1]))))
    return points[indices]


def _edge_lengths(points: jnp.ndarray) -> list[float]:
    if points.shape[0] < 2:
        return []
    lengths = [
        float(jnp.linalg.norm(points[i] - points[j]))
        for i, j in combinations(range(points.shape[0]), 2)
    ]
    return sorted(lengths)


@pytest.mark.goal_math
@pytest.mark.deep
@settings(max_examples=5, deadline=None)
@given(
    st.integers(min_value=6, max_value=10),
    st.integers(min_value=0, max_value=2**31 - 1),
    st.floats(min_value=0.1, max_value=1.5),
)
def test_capacity_and_volume_affine_symplectic_invariance(
    n_vertices: int, seed: int, translation_scale: float
) -> None:
    """Every capacity/volume algorithm is invariant under affine symplectomorphisms."""

    vertices = _random_convex_hull_vertices(seed, n_vertices)
    bundle = polytopes.build_from_vertices(vertices)
    assume(bundle.vertices.shape[0] >= 5)

    key = jax.random.PRNGKey(seed)
    matrix = symplectic.random_symplectic_matrix(key, bundle.dimension)
    translation = jnp.asarray(
        np.random.default_rng(seed + 1).normal(size=(bundle.dimension,)) * translation_scale,
        dtype=jnp.float64,
    )
    transformed_vertices = vertices @ matrix.T + translation
    transformed_bundle = polytopes.build_from_vertices(transformed_vertices)
    assume(transformed_bundle.vertices.shape[0] >= 5)

    for name, evaluator in _capacity_evaluators_for_invariance():
        try:
            baseline = evaluator(bundle)
            transformed = evaluator(transformed_bundle)
        except ValueError:
            assume(False)
        else:
            assert jnp.isclose(baseline, transformed, rtol=1e-6, atol=1e-9), name

    for name, evaluator in _volume_evaluators():
        baseline = evaluator(bundle)
        transformed = evaluator(transformed_bundle)
        assert jnp.isclose(baseline, transformed, rtol=1e-9, atol=1e-12), name


@pytest.mark.goal_math
@pytest.mark.smoke
@pytest.mark.parametrize("name, evaluator", _capacity_evaluators_for_invariance())
def test_capacity_scaling_law_on_simplex(name: str, evaluator: CapacityEvaluator) -> None:
    """Capacity scales quadratically under linear scaling for every solver."""

    base = polytopes.build_from_vertices(_simplex_vertices(edge=2.0))
    scaled = polytopes.build_from_vertices(base.vertices * 1.7)
    baseline = evaluator(base)
    scaled_value = evaluator(scaled)
    expected = (1.7**2) * baseline
    assert scaled_value == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
@pytest.mark.parametrize("name, evaluator", _volume_evaluators())
def test_volume_scaling_law_on_simplex(name: str, evaluator: VolumeEvaluator) -> None:
    """Volume scales with exponent 2n under linear scaling."""

    base = polytopes.build_from_vertices(_simplex_vertices(edge=2.0))
    scaled = polytopes.build_from_vertices(base.vertices * 1.7)
    baseline = evaluator(base)
    scaled_value = evaluator(scaled)
    # 4D simplex → n = 2.
    expected = (1.7**4) * baseline
    assert scaled_value == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_cycle_equivariance_under_affine_symplectomorphism() -> None:
    """Minimum cycles transform equivariantly under affine symplectomorphisms."""

    bundle = polytopes.build_from_vertices(_simplex_vertices(edge=2.0))
    original_cycle = minimum_cycle_reference(bundle)
    assert original_cycle.shape[0] > 0

    matrix = jnp.eye(bundle.dimension, dtype=jnp.float64)
    translation = jnp.asarray([0.3, -0.5, 0.2, 0.1], dtype=jnp.float64)

    transformed_vertices = bundle.vertices @ matrix.T + translation
    transformed_bundle = polytopes.build_from_vertices(transformed_vertices)
    transformed_cycle = minimum_cycle_reference(transformed_bundle)

    expected_cycle = original_cycle @ matrix.T + translation
    assert transformed_cycle.shape == expected_cycle.shape
    assert _edge_lengths(transformed_cycle) == pytest.approx(
        _edge_lengths(expected_cycle), rel=1e-9, abs=1e-12
    )


@pytest.mark.goal_math
@pytest.mark.smoke
@pytest.mark.parametrize("name, evaluator", _capacity_evaluators_simplex())
def test_capacity_simplex_matches_by_hand_value(name: str, evaluator: CapacityEvaluator) -> None:
    """Every exact solver reproduces the unit value for the canonical 4D simplex."""

    bundle = polytopes.build_from_vertices(_simplex_vertices(edge=2.0))
    value = evaluator(bundle)
    assert value == pytest.approx(1.0, rel=1e-12, abs=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
@pytest.mark.parametrize("name, evaluator", _volume_evaluators())
def test_cube_volume_matches_closed_form(name: str, evaluator: VolumeEvaluator) -> None:
    """Hypercube volume matches the analytic formula (2r)^{2n}."""

    cube = hypercube(4, radius=1.25).geometry
    expected = (2.0 * 1.25) ** 4
    value = evaluator(cube)
    assert value == pytest.approx(expected, rel=1e-12, abs=1e-8)


__all__ = [
    "test_capacity_and_volume_affine_symplectic_invariance",
    "test_capacity_scaling_law_on_simplex",
    "test_volume_scaling_law_on_simplex",
    "test_cycle_equivariance_under_affine_symplectomorphism",
    "test_capacity_simplex_matches_by_hand_value",
    "test_cube_volume_matches_closed_form",
]
