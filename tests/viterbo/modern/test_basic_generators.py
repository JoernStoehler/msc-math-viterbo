"""Basic generator semantics against the modern Polytope type.

Covers the intention that generators produce `Polytope` samples with expected
dimensions and float64 data. These tests currently fail until the generator
implementations are completed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from viterbo.modern import basic_generators
from viterbo.modern.types import Polytope


@pytest.mark.goal_code
@pytest.mark.smoke
def test_sample_uniform_ball_produces_polytopes() -> None:
    """Uniform-ball generator returns a list of `Polytope` with correct shapes."""
    key = jax.random.PRNGKey(0)
    dimension = 3
    num_samples = 2
    samples = basic_generators.sample_uniform_ball(key, dimension, num_samples=num_samples)
    assert isinstance(samples, list)
    assert len(samples) == num_samples
    for poly in samples:
        assert isinstance(poly, Polytope)
        assert poly.vertices.shape[1] == dimension
        assert poly.vertices.dtype == jnp.float64


@pytest.mark.goal_code
@pytest.mark.smoke
def test_sample_uniform_sphere_vertices_on_unit_sphere() -> None:
    """Uniform-sphere generator returns vertices with unit-norm rows (within tol)."""
    key = jax.random.PRNGKey(42)
    dimension = 3
    num_samples = 2
    samples = basic_generators.sample_uniform_sphere(key, dimension, num_samples=num_samples)
    for poly in samples:
        norms = jnp.linalg.norm(poly.vertices, axis=-1)
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
    for poly in samples:
        assert poly.offsets.shape == (num_facets,)
        assert jnp.allclose(poly.offsets, 1.0, rtol=0.0, atol=0.0)
