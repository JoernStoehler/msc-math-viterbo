from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.exp1.halfspaces import enumerate_vertices


@dataclass(frozen=True)
class NormalCone:
    vertex: Float[Array, " dim"]
    active_facets: tuple[int, ...]


@dataclass(frozen=True)
class MinkowskiNormalFan:
    vertices: Float[Array, " k dim"]
    cones: tuple[NormalCone, ...]
    neighbors: tuple[tuple[int, ...], ...]

    @property
    def dimension(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


def build_normal_fan(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    atol: float = 1e-9,
) -> MinkowskiNormalFan:
    """Construct a normal-fan-like structure from a half-space system.

    Approximates normal cones by taking active facets at each enumerated vertex
    and connecting vertices whose active sets share at least dim-1 facets.
    """
    A = jnp.asarray(normals, dtype=jnp.float64)
    b = jnp.asarray(offsets, dtype=jnp.float64)
    verts = enumerate_vertices(A, b, atol=atol)
    dim = int(A.shape[1])

    cones: list[NormalCone] = []
    for k in range(int(verts.shape[0])):
        x = verts[k]
        residuals = A @ x - b
        active = jnp.where(jnp.abs(residuals) <= float(atol))[0]
        cones.append(NormalCone(vertex=x, active_facets=tuple(int(i) for i in active.tolist())))

    n = len(cones)
    adj = jnp.zeros((n, n), dtype=bool)
    for i in range(n):
        fi = set(cones[i].active_facets)
        for j in range(i + 1, n):
            fj = set(cones[j].active_facets)
            if len(fi.intersection(fj)) >= max(0, dim - 1):
                adj = adj.at[i, j].set(True)
                adj = adj.at[j, i].set(True)
    neighbors = tuple(
        tuple(int(idx) for idx in jnp.where(adj[row])[0].tolist()) for row in range(n)
    )
    return MinkowskiNormalFan(vertices=verts, cones=tuple(cones), neighbors=neighbors)
