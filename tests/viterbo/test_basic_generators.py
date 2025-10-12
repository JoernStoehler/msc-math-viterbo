"""Basic generator semantics against the modern generators API (datasets2).

Covers that generators produce samples with expected dimensions and float64 data.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from viterbo.datasets2 import generators as basic_generators


@pytest.mark.goal_code
@pytest.mark.smoke
def test_sample_uniform_ball_produces_polytopes() -> None:
    """Uniform-ball generator returns a list of samples with correct shapes."""
    key = jax.random.PRNGKey(0)
    dimension = 3
    num_samples = 2
    samples = basic_generators.sample_uniform_ball(key, dimension, num_samples=num_samples)
    assert isinstance(samples, (tuple, list))
    assert len(samples) == num_samples
    for sample in samples:
        assert sample.vertices.shape[1] == dimension
        assert sample.vertices.dtype == jnp.float64


@pytest.mark.goal_code
@pytest.mark.smoke
def test_sample_uniform_sphere_vertices_on_unit_sphere() -> None:
    """Uniform-sphere generator returns vertices with unit-norm rows (within tol)."""
    key = jax.random.PRNGKey(42)
    dimension = 3
    num_samples = 2
    samples = basic_generators.sample_uniform_sphere(key, dimension, num_samples=num_samples)
    for sample in samples:
        norms = jnp.linalg.norm(sample.vertices, axis=-1)
        assert jnp.allclose(norms, 1.0, rtol=1e-12, atol=1e-12)


@pytest.mark.goal_code
@pytest.mark.smoke
def test_sample_halfspace_tangent_offsets_equal_one() -> None:
    """Halfspace-tangent generator sets all offsets to 1.0 by construction."""
    key = jax.random.PRNGKey(7)
    dimension = 2
    num_facets = 4
    num_samples = 2
    samples = basic_generators.sample_halfspace_tangent(
        key, dimension, num_facets=num_facets, num_samples=num_samples
    )
    for sample in samples:
        assert sample.offsets.shape == (num_facets,)
        assert jnp.allclose(sample.offsets, 1.0, rtol=0.0, atol=0.0)
