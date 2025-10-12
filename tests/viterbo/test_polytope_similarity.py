"""Unit tests for modern symplectic polytope similarity primitives (stubs)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from viterbo.polytopes_similarity import (
    CorrelationSketch,
    JaccardMonteCarloCache,
    RadialProfileCache,
    SymplecticSpectrumCache,
    build_correlation_sketch,
    build_jaccard_cache,
    build_radial_profile_cache,
    centre_polytope_vertices,
    radial_profile_distance,
    radial_profile_distance_cached,
    staged_symplectic_similarity,
    symplectic_correlation_distance,
    symplectic_correlation_distance_cached,
    symplectic_jaccard_distance,
    symplectic_jaccard_distance_cached,
    symplectic_spectrum_from_covariance,
)
from viterbo.datasets.types import Polytope


def _unit_square() -> Polytope:
    normals = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    offsets = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
    vertices = jnp.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=jnp.float64,
    )
    incidence = jnp.array(
        [
            [True, False, True, False],
            [False, True, True, False],
            [False, True, False, True],
            [True, False, False, True],
        ],
        dtype=bool,
    )
    return Polytope(normals=normals, offsets=offsets, vertices=vertices, incidence=incidence)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_centre_polytope_vertices_translation_to_origin() -> None:
    """Recentering should translate the unit square so its centroid matches the origin."""

    polytope = _unit_square()
    translation = centre_polytope_vertices(polytope)
    recentred = polytope.vertices - translation
    mean = jnp.mean(recentred, axis=0)
    np.testing.assert_allclose(np.asarray(mean), np.zeros(2), rtol=1e-12, atol=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_symplectic_spectrum_from_covariance_planar_formula() -> None:
    """Symplectic eigenvalue in two dimensions equals sqrt(det(covariance))."""

    covariance = jnp.array([[3.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    expected = math.sqrt(float(np.linalg.det(np.asarray(covariance))))
    spectrum = symplectic_spectrum_from_covariance(covariance)
    assert spectrum.shape == (1,)
    np.testing.assert_allclose(np.asarray(spectrum), np.array([expected]), rtol=1e-12, atol=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_radial_profile_distance_translation_invariance() -> None:
    """Radial distance should coincide for a square and its translation after centring."""

    square = _unit_square()
    translation = jnp.array([2.5, -3.0], dtype=jnp.float64)
    translated = Polytope(
        normals=square.normals,
        offsets=square.offsets + square.normals @ translation,
        vertices=square.vertices + translation,
        incidence=square.incidence,
    )
    directions = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
        ],
        dtype=jnp.float64,
    )
    distance = radial_profile_distance(
        square,
        translated,
        directions=directions,
        softness=0.05,
        epsilon=1e-9,
    )
    assert math.isclose(distance, 0.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_symplectic_correlation_distance_identical_polytopes() -> None:
    """Correlation distance must vanish when both polytopes coincide."""

    square = _unit_square()
    histogram_edges = jnp.linspace(-2.0, 2.0, num=65, dtype=jnp.float64)
    distance = symplectic_correlation_distance(
        square,
        square,
        num_pairs=2000,
        histogram_edges=histogram_edges,
        seed=0,
    )
    assert math.isclose(distance, 0.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_math
@pytest.mark.smoke
def test_symplectic_jaccard_distance_identical_polytopes() -> None:
    """Symplectic Jaccard distance evaluates to zero on identical polytopes."""

    square = _unit_square()
    distance = symplectic_jaccard_distance(
        square,
        square,
        num_samples=4000,
        num_restarts=2,
        num_iterations=50,
        search_learning_rate=0.05,
        seed=0,
    )
    assert math.isclose(distance, 0.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_cached_and_direct_paths_align() -> None:
    """Direct and cached paths should deliver matching distances for identical inputs."""

    square = _unit_square()
    directions = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
        ],
        dtype=jnp.float64,
    )
    direct_distance = radial_profile_distance(
        square,
        square,
        directions=directions,
        softness=0.05,
        epsilon=1e-9,
    )
    cache_a = build_radial_profile_cache(
        square,
        directions=directions,
        softness=0.05,
        epsilon=1e-9,
    )
    cache_b = build_radial_profile_cache(
        square,
        directions=directions,
        softness=0.05,
        epsilon=1e-9,
    )
    cached_distance = radial_profile_distance_cached(cache_a, cache_b)
    assert isinstance(cache_a, RadialProfileCache)
    assert isinstance(cache_b, RadialProfileCache)
    assert math.isclose(direct_distance, cached_distance, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_staged_similarity_combines_caches() -> None:
    """Staged similarity should honour weighting thresholds using cached artefacts."""

    zero_radial = RadialProfileCache(
        log_radii=jnp.zeros(3, dtype=jnp.float64),
        directions=jnp.eye(3, dtype=jnp.float64),
        softness=0.05,
        epsilon=1e-9,
    )
    zero_correlation = CorrelationSketch(
        samples=jnp.zeros(4, dtype=jnp.float64),
        histogram=jnp.zeros(3, dtype=jnp.float64),
        bin_edges=jnp.zeros(4, dtype=jnp.float64),
        num_pairs=4,
    )
    zero_spectrum = SymplecticSpectrumCache(
        centred_covariance=jnp.eye(2, dtype=jnp.float64),
        williamson_transform=jnp.eye(2, dtype=jnp.float64),
        symplectic_eigenvalues=jnp.ones(1, dtype=jnp.float64),
    )
    score = staged_symplectic_similarity(
        zero_spectrum,
        zero_spectrum,
        zero_radial,
        zero_radial,
        zero_correlation,
        zero_correlation,
        weights=jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64),
        near_threshold=0.1,
        far_threshold=0.9,
    )
    assert math.isclose(score, 0.0, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_cached_jaccard_matches_direct_parameters() -> None:
    """Cached Jaccard artefacts should reproduce direct distance settings."""

    square = _unit_square()
    cache = build_jaccard_cache(square, num_samples=4000, seed=0)
    assert isinstance(cache, JaccardMonteCarloCache)
    distance = symplectic_jaccard_distance_cached(
        cache,
        cache,
        num_restarts=2,
        num_iterations=50,
        search_learning_rate=0.05,
    )
    direct = symplectic_jaccard_distance(
        square,
        square,
        num_samples=4000,
        num_restarts=2,
        num_iterations=50,
        search_learning_rate=0.05,
        seed=0,
    )
    assert math.isclose(distance, direct, rel_tol=1e-12, abs_tol=0.0)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_cached_correlation_matches_direct_parameters() -> None:
    """Correlation sketches should agree with direct estimators when reused."""

    square = _unit_square()
    histogram_edges = jnp.linspace(-2.0, 2.0, num=65, dtype=jnp.float64)
    sketch = build_correlation_sketch(
        square,
        num_pairs=2000,
        histogram_edges=histogram_edges,
        seed=0,
    )
    assert isinstance(sketch, CorrelationSketch)
    cached = symplectic_correlation_distance_cached(sketch, sketch)
    direct = symplectic_correlation_distance(
        square,
        square,
        num_pairs=2000,
        histogram_edges=histogram_edges,
        seed=0,
    )
    assert math.isclose(cached, direct, rel_tol=1e-12, abs_tol=0.0)
