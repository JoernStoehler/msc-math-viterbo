"""Polytope generators for the modern API."""

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from viterbo.modern.polytopes import build_from_halfspaces, build_from_vertices
from viterbo.modern.types import Polytope

def sample_halfspace(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_facets: int,
    num_samples: int,
) -> list[Polytope]:
    """Yield polytopes sampled from the halfspace representation.
    
    normal ~ Uniform(S^{dimension-1})
    offset ~ Uniform[0.5, 2]
    
    We reject unbounded samples.
    """
    key1, key2 = jax.random.split(key, 2)
    normals = jax.random.normal(key1, (num_samples, num_facets, dimension), dtype=jnp.float64)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    offsets = jax.random.uniform(
        key2, (num_samples, num_facets), minval=0.5, maxval=2.0, dtype=jnp.float64
    )
    return [build_from_halfspaces(normals=normals[i], offsets=offsets[i]) for i in range(normals.shape[0])]

def sample_halfspace_tangent(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_facets: int,
    num_samples: int,
) -> list[Polytope]:
    """Yield polytopes sampled from the halfspace representation, tangent to the unit ball.
    
    normal ~ Uniform(S^{dimension-1})
    offset = 1
    
    We reject unbounded samples.
    """
    normals = jax.random.normal(key, (num_samples, num_facets, dimension), dtype=jnp.float64)
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
    offsets = jnp.ones((num_samples, num_facets), dtype=jnp.float64)
    return [build_from_halfspaces(normals=normals[i], offsets=offsets[i]) for i in range(normals.shape[0])]

def sample_uniform_sphere(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_samples: int,
) -> list[Polytope]:
    """Yield polytopes sampled from the unit sphere.

    vertices ~ Uniform(S^{dimension-1})
    We reject samples that do not contain the origin in their interior.
    """
    num_vertices = dimension + 1
    ys = jax.random.normal(key, (num_samples, num_vertices, dimension), dtype=jnp.float64)
    vertices = ys / jnp.linalg.norm(ys, axis=-1, keepdims=True)
    return [build_from_vertices(vertices=vertices[i]) for i in range(vertices.shape[0])]

def sample_uniform_ball(
    key: PRNGKeyArray,
    dimension: int,
    *,
    num_samples: int,
) -> list[Polytope]:
    """Yield polytopes sampled from the unit ball.

    vertices ~ Uniform(B^{dimension-1})
    We reject samples that do not contain the origin in their interior.
    """
    key1, key2 = jax.random.split(key, 2)
    num_vertices = dimension + 1
    ys = jax.random.normal(key1, (num_samples, num_vertices, dimension), dtype=jnp.float64)
    dirs = ys / jnp.linalg.norm(ys, axis=-1, keepdims=True)
    u = jax.random.uniform(key2, (num_samples, num_vertices, 1), dtype=jnp.float64)
    radii = jnp.power(u, 1.0 / dimension)
    vertices = dirs * radii
    return [build_from_vertices(vertices=vertices[i]) for i in range(vertices.shape[0])]

def enumerate_product_ngons(
    max_ngon_P: int,
    max_ngon_Q: int,
    max_rotation_Q: int,
) -> list[Polytope]:
    """Enumerate products of two regular polygons.

    We rotate the k_Q-gon by 2 * pi * r/s for coprime r,s <= max_rotation_Q and r/s < 1/k_Q.
    """
    samples = []
    for k_P in range(3, max_ngon_P + 1):
        for k_Q in range(3, max_ngon_Q + 1):
            for s in range(1, max_rotation_Q + 1):
                for r in range(0, s):
                    if r == 0 and s != 1:
                        continue
                    if r / s >= 1 / k_Q:
                        continue
                    if r != 0 and np.gcd(r, s) != 1:
                        continue

                    angle = 2 * np.pi * r / s
                    # build vertices
                    vertices_P = jnp.array([
                        [np.cos(2 * np.pi * i / k_P), np.sin(2 * np.pi * i / k_P)]
                        for i in range(k_P)
                    ], dtype=jnp.float64)
                    vertices_Q = jnp.array([
                        [np.cos(2 * np.pi * i / k_Q + angle), np.sin(2 * np.pi * i / k_Q + angle)]
                        for i in range(k_Q)
                    ], dtype=jnp.float64)
                    # form product
                    vertices = jnp.array([
                        jnp.concatenate([v_P, v_Q])
                        for v_P in vertices_P
                        for v_Q in vertices_Q
                    ], dtype=jnp.float64)
                    samples.append(build_from_vertices(vertices=vertices))
    return samples
