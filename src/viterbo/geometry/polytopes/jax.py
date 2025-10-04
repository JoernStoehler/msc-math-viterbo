"""JAX-backed polytope combinatorics leveraging the JAX half-space solvers."""

from __future__ import annotations

from viterbo.geometry.halfspaces import jax as halfspaces_jax
from viterbo.geometry.polytopes import _shared
from viterbo.geometry.polytopes import reference as _reference

Polytope = _shared.Polytope
PolytopeCombinatorics = _shared.PolytopeCombinatorics

vertices_from_halfspaces = halfspaces_jax.enumerate_vertices
halfspaces_from_vertices = _reference.halfspaces_from_vertices


def polytope_combinatorics(
    polytope: Polytope,
    *,
    atol: float = 1e-9,
    use_cache: bool = True,
) -> PolytopeCombinatorics:
    """Compute combinatorics using ``jax.numpy`` linear algebra kernels."""
    key = _shared.polytope_cache_key(polytope, atol)
    if _shared.cache_enabled(use_cache):
        cached = _shared.cache_lookup(key)
        if cached is not None:
            return cached

    B, c = polytope.halfspace_data()
    vertices = halfspaces_jax.enumerate_vertices(B, c, atol=atol)
    combinatorics = _shared.build_combinatorics(B, c, vertices, atol=atol)

    if _shared.cache_enabled(use_cache):
        _shared.cache_store(key, combinatorics)

    return combinatorics
