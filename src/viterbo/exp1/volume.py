from __future__ import annotations

import math
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo._wrapped.spatial import QhullError, convex_hull_volume, delaunay_simplices
from viterbo.exp1.halfspaces import enumerate_vertices as _enumerate_vertices
from viterbo.exp1.polytopes import (
    HalfspacePolytope,
    LagrangianProductPolytope,
    Polytope,
    VertexPolytope,
    to_halfspaces,
)



def _halfspaces_for(P: Polytope) -> tuple[Float[Array, " m dim"], Float[Array, " m"]]:
    if isinstance(P, HalfspacePolytope):
        return P.as_tuple()
    if isinstance(P, VertexPolytope):
        (verts,) = P.as_tuple()
        from viterbo.exp1.halfspaces import halfspaces_from_vertices as _hfv

        normals, offsets = _hfv(verts)
        return normals, offsets
    if isinstance(P, LagrangianProductPolytope):
        normals_left = jnp.hstack((P.normals_p, jnp.zeros((P.normals_p.shape[0], 2))))
        normals_right = jnp.hstack((jnp.zeros((P.normals_q.shape[0], 2)), P.normals_q))
        normals = jnp.vstack((normals_left, normals_right))
        offsets = jnp.concatenate((P.offsets_p, P.offsets_q))
        return normals, offsets
    raise TypeError("Unsupported polytope variant")


def volume(
    P: Polytope,
    *,
    method: Literal["triangulation", "fast", "monte_carlo"] = "triangulation",
    num_samples: int | None = None,
) -> Float[Array, ""]:
    """Compute the Euclidean volume using the selected strategy.

    Semantics:
      - "triangulation" uses a trusted hull-based estimator (reference).
      - "fast" uses Delaunay simplices with hull fallback.
      - "monte_carlo" performs rejection sampling in a bounding box.
    Returns a JAX scalar ``Float[Array, ""]``.
    """
    normals, offsets = _halfspaces_for(P)
    if method == "triangulation":
        verts = _enumerate_vertices(normals, offsets)
        val = convex_hull_volume(verts, qhull_options="QJ")
        return jnp.asarray(val, dtype=jnp.float64)
    if method == "fast":
        verts = _enumerate_vertices(normals, offsets)
        try:
            simplices = delaunay_simplices(verts, qhull_options="QJ")
            val = _volume_of_simplices(verts[simplices])
        except QhullError:
            val = convex_hull_volume(verts, qhull_options="QJ")
        return jnp.asarray(val, dtype=jnp.float64)
    if method == "monte_carlo":
        normals, offsets = to_halfspaces(P).as_tuple()
        return volume_monte_carlo(normals, offsets, num_samples=num_samples or 100_000)
    raise ValueError("Unknown method for volume")


def volume_triangulation(verts: Float[Array, " k dim"]) -> Float[Array, ""]:
    """Trusted hull-based volume estimator from a vertex cloud."""
    val = convex_hull_volume(jnp.asarray(verts, dtype=jnp.float64), qhull_options="QJ")
    return jnp.asarray(val, dtype=jnp.float64)


def volume_tria_fast(verts: Float[Array, " k dim"]) -> Float[Array, ""]:
    """Fast volume estimator via Delaunay simplices with hull fallback."""
    verts = jnp.asarray(verts, dtype=jnp.float64)
    try:
        simplices = delaunay_simplices(verts, qhull_options="QJ")
        val = _volume_of_simplices(verts[simplices])
    except QhullError:
        val = convex_hull_volume(verts, qhull_options="QJ")
    return jnp.asarray(val, dtype=jnp.float64)


def volume_monte_carlo(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    num_samples: int = 100_000,
) -> Float[Array, ""]:
    """Approximate volume via rejection sampling inside an axis-aligned box.

    Notes:
      - Suited for demos only; accuracy scales slowly with samples.
      - Deterministic seed for reproducibility in examples/tests.
    """
    # Simple rejection sampling in an axis-aligned bounding box from vertices.
    verts = _enumerate_vertices(
        jnp.asarray(normals, dtype=jnp.float64), jnp.asarray(offsets, dtype=jnp.float64)
    )
    mins = jnp.min(verts, axis=0)
    maxs = jnp.max(verts, axis=0)
    box_volume = float(jnp.prod(maxs - mins))
    if not math.isfinite(box_volume) or box_volume <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    key = jax.random.PRNGKey(0)
    dim = int(verts.shape[1])
    samples = (
        jax.random.uniform(key, (int(num_samples), dim), dtype=jnp.float64) * (maxs - mins) + mins
    )
    inside = (samples @ jnp.asarray(normals).T) <= jnp.asarray(offsets)
    counts = jnp.all(inside, axis=1).sum()
    ratio = counts / float(num_samples)
    return jnp.asarray(box_volume * ratio, dtype=jnp.float64)


def _volume_of_simplices(stacks: Float[Array, " n (dim+1) dim"]) -> float:
    """Return the total volume of a stack of simplices.

    Each simplex is a (dim+1)×dim set of vertices; volume is |det(V1-V0,...,Vd-V0)|/d!.
    """
    stacks = jnp.asarray(stacks, dtype=jnp.float64)
    n = int(stacks.shape[0])
    dim = int(stacks.shape[2])
    if n == 0:
        return 0.0
    base = stacks[:, 0:1, :]
    edges = stacks[:, 1:, :] - base
    # Compute determinants for each simplex
    dets = jnp.linalg.det(edges)
    vols = jnp.abs(dets) / math.factorial(dim)
    return float(jnp.sum(vols))
