from __future__ import annotations

from jaxtyping import Array, Float

from viterbo.exp1.reeb_cycles.graph import build_oriented_edge_graph
from viterbo.exp1.capacity_ehz import capacity_halfspace_optimization


def _find_simple_cycle(graph: dict[int, set[int]]) -> list[int] | None:
    """Return a simple directed cycle as a list of edge ids, if one exists."""
    visited: set[int] = set()
    stack: list[int] = []
    in_stack: set[int] = set()

    def dfs(u: int) -> list[int] | None:
        visited.add(u)
        stack.append(u)
        in_stack.add(u)
        for v in sorted(graph.get(u, set())):
            if v not in visited:
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif v in in_stack:
                if v in stack:
                    idx = stack.index(v)
                    return stack[idx:] + [v]
        stack.pop()
        in_stack.discard(u)
        return None

    for start in list(graph.keys()):
        if start not in visited:
            cyc = dfs(start)
            if cyc is not None:
                return cyc
    return None


def compute_ehz_capacity_reference(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    atol: float = 1e-9,
) -> float:
    """Compute c_EHZ by validating the oriented-edge graph and calling facet solver.

    This is a starting point for 4D Reeb-cycle support in exp1. It validates that the
    oriented-edge graph is non-empty (admissible edges exist) and delegates to the
    facet-normal reference solver for the capacity value.
    """
    graph = build_oriented_edge_graph(normals, offsets, atol=atol)
    if len(graph.edges) == 0:
        raise ValueError("Oriented-edge graph is empty; polytope lacks admissible edges.")
    return float(capacity_halfspace_optimization(normals, offsets, tol=atol))


def compute_ehz_capacity_and_cycle_reference(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    atol: float = 1e-9,
) -> tuple[float, Float[Array, " n dim"]]:
    """Return capacity and one admissible Reeb cycle as 4D points (dim=4 required).

    The cycle extraction currently returns the first simple cycle found in the oriented-edge
    graph; future work can prioritize cycles that realize the capacity.
    """
    og = build_oriented_edge_graph(normals, offsets, atol=atol)
    if len(og.edges) == 0:
        raise ValueError("Oriented-edge graph is empty; polytope lacks admissible edges.")
    cap = float(capacity_halfspace_optimization(normals, offsets, tol=atol))
    cyc_ids = _find_simple_cycle(og.graph)
    if cyc_ids is None or len(cyc_ids) < 3:
        raise ValueError("Failed to extract a simple Reeb cycle.")
    verts = og.vertices
    ids = cyc_ids[:-1]
    pts = [verts[og.edges[eid].tail_vertex] for eid in ids]
    import jax.numpy as jnp

    cycle_points = jnp.stack(pts, axis=0) if pts else jnp.zeros((0, 4), dtype=jnp.float64)
    return cap, cycle_points
