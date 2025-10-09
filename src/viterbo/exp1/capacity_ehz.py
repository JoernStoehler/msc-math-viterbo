from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from viterbo.exp1.polytopes import (
    HalfspacePolytope,
    LagrangianProductPolytope,
    Polytope,
    VertexPolytope,
    to_halfspaces,
)
from viterbo.exp1.minkowski_billiards.fan import build_normal_fan as _build_fan
from viterbo.exp1.minkowski_billiards.reference import cycle_length, enumerate_cycles


def capacity(P: Polytope, method: str = "auto", *, geometry: LagrangianProductPolytope | None = None, max_bounces: int | None = None, tol: float = 1e-10) -> Float[Array, ""]:
    """Compute capacity via Minkowski or facet-normal (exp1 scope).

    - "minkowski_reference": 2×2 product only; returns Minkowski billiard length.
    - "facet_reference": Haim–Kislev facet-normal formula (small dims; m<=7).
    - "auto": choose Minkowski for Lagrangian products; otherwise facet_reference.
    """
    if method.startswith("minkowski") or (method == "auto" and isinstance(P, LagrangianProductPolytope)):
        if not isinstance(P, LagrangianProductPolytope):
            raise ValueError("Minkowski method requires a LagrangianProductPolytope")
        val, _ = capacity_minkowski_billiard(P, geometry=geometry, max_bounces=max_bounces, tol=tol)
        return jnp.asarray(val, dtype=jnp.float64)
    if method in ("facet_reference", "auto"):
        if isinstance(P, LagrangianProductPolytope):
            # Embedded half-spaces in R^4
            a1_pad = jnp.hstack((P.normals_p, jnp.zeros((P.normals_p.shape[0], 2))))
            a2_pad = jnp.hstack((jnp.zeros((P.normals_q.shape[0], 2)), P.normals_q))
            normals = jnp.vstack((a1_pad, a2_pad))
            offsets = jnp.concatenate((P.offsets_p, P.offsets_q))
        elif isinstance(P, HalfspacePolytope):
            normals, offsets = P.as_tuple()
        elif isinstance(P, VertexPolytope):
            hp = to_halfspaces(P)
            normals, offsets = hp.as_tuple()
        else:
            raise ValueError("Facet method requires a half-space or vertex representation.")
        return capacity_halfspace_optimization(normals, offsets, tol=tol)
    raise ValueError("Unknown method for capacity")


def capacity_and_cycle(
    P: Polytope,
    *,
    method: str = "minkowski_reference",
    geometry: LagrangianProductPolytope | None = None,
    max_bounces: int | None = None,
    tol: float = 1e-9,
) -> tuple[Float[Array, ""], Float[Array, " n dim"]]:
    """Return capacity and a realizing cycle for supported methods.

    Currently implemented for the Minkowski billiard on 2×2 products, where
    the cycle is returned as points in ``R^4``.
    """
    if method.startswith("minkowski"):
        if not isinstance(P, LagrangianProductPolytope):
            raise ValueError("Minkowski method requires a LagrangianProductPolytope")
        return capacity_minkowski_billiard(P, geometry=geometry, max_bounces=max_bounces, tol=tol)
    raise NotImplementedError("capacity_and_cycle currently implemented for Minkowski method only")


def capacity_halfspace_optimization(
    A: Float[Array, " m dim"],
    b: Float[Array, " m"],
    *,
    tol: float = 1e-10,
) -> Float[Array, ""]:
    """Facet-normal EHZ capacity via Haim–Kislev (small dims, reference).

    Notes:
      - Enumerates subsets of size dim+1; uses permutation search for m<=7.
      - For larger subsets, raises NotImplementedError in exp1.
    """
    B = jnp.asarray(A, dtype=jnp.float64)
    c = jnp.asarray(b, dtype=jnp.float64)
    if B.ndim != 2 or c.ndim != 1 or B.shape[0] != c.shape[0]:
        raise ValueError("Invalid half-space system")
    dim = int(B.shape[1])
    if dim % 2 != 0 or dim < 2:
        raise ValueError("Ambient dimension must be 2n, n>=1")
    subset_size = dim + 1
    J = _standard_symplectic_matrix(dim)
    best = jnp.inf
    mfacets = int(B.shape[0])
    for rows in itertools.combinations(range(mfacets), subset_size):
        beta, W = _prepare_subset(B, c, rows, J, tol=tol)
        if beta is None or W is None:
            continue
        candidate = _subset_capacity_candidate(beta, W, tol=tol)
        if candidate is None:
            continue
        if candidate < best:
            best = candidate
    if not bool(jnp.isfinite(best)):
        raise ValueError("No admissible facet subset satisfied constraints.")
    return jnp.asarray(float(best), dtype=jnp.float64)


def _product_vertices(P: LagrangianProductPolytope) -> Float[Array, " k 4"]:
    """Return Cartesian product vertices in R^4 via a simple Python loop."""
    left = jnp.asarray(P.verts_p, dtype=jnp.float64)
    right = jnp.asarray(P.verts_q, dtype=jnp.float64)
    pts: list[Array] = []
    for i in range(int(left.shape[0])):
        for j in range(int(right.shape[0])):
            pts.append(jnp.concatenate((left[i], right[j])))
    if not pts:
        return jnp.zeros((0, 4), dtype=jnp.float64)
    return jnp.stack(pts, axis=0)


def capacity_reeb_orbits(
    A: Float[Array, " m 4"],
    b: Float[Array, " m"],
) -> tuple[Float[Array, ""], Float[Array, " n 4"]]:
    """Placeholder: returns reference capacity; cycle extraction TBD for 4D."""
    from viterbo.exp1.reeb_cycles.reference import (
        compute_ehz_capacity_and_cycle_reference as _reeb_cap_cyc,
    )

    cap, pts = _reeb_cap_cyc(A, b, atol=1e-9)
    return jnp.asarray(cap, dtype=jnp.float64), jnp.asarray(pts, dtype=jnp.float64)


def capacity_minkowski_billiard(
    P: LagrangianProductPolytope,
    *,
    geometry: LagrangianProductPolytope | None = None,
    max_bounces: int | None = None,
    tol: float = 1e-9,
) -> tuple[Float[Array, ""], Float[Array, " n 4"]]:
    """Minimal closed (K,T)-Minkowski billiard length and realizing cycle.

    Computes the normal fan on the billiard table and enumerates simple cycles
    (reference). Returns the minimal length and the corresponding 4D cycle.
    """
    A1_pad = jnp.hstack((P.normals_p, jnp.zeros((P.normals_p.shape[0], 2))))
    A2_pad = jnp.hstack((jnp.zeros((P.normals_q.shape[0], 2)), P.normals_q))
    A_table = jnp.vstack((A1_pad, A2_pad))
    b_table = jnp.concatenate((P.offsets_p, P.offsets_q))
    fan = _build_fan(A_table, b_table, atol=tol)
    geom_vertices = _product_vertices(P if geometry is None else geometry)
    best = jnp.inf
    best_cycle: tuple[int, ...] | None = None
    for cycle in enumerate_cycles(fan, max_length=max_bounces or (fan.dimension + 2)):
        length = cycle_length(cycle, fan.vertices, geom_vertices)
        if length < best:
            best = length
            best_cycle = cycle
    if best_cycle is None or not bool(jnp.isfinite(best)):
        raise ValueError("No closed Minkowski billiard cycle satisfies the constraints.")
    # Assemble 4D cycle points from 2D vertex factors: points are fan.vertices in R^4
    indices = jnp.asarray(best_cycle, dtype=int)
    cycle_points = fan.vertices[indices]
    return jnp.asarray(best, dtype=jnp.float64), jnp.asarray(cycle_points, dtype=jnp.float64)


def _standard_symplectic_matrix(dimension: int) -> Float[Array, " dim dim"]:
    if dimension % 2 != 0:
        raise ValueError("Symplectic matrix requires even dimension")
    n = dimension // 2
    upper = jnp.hstack((jnp.zeros((n, n)), -jnp.eye(n)))
    lower = jnp.hstack((jnp.eye(n), jnp.zeros((n, n))))
    j_symp = jnp.vstack((upper, lower))
    return jnp.asarray(j_symp, dtype=jnp.float64)


def _prepare_subset(
    B: Float[Array, " M D"],
    c: Float[Array, " M"],
    rows: tuple[int, ...] | list[int] | range,
    J: Float[Array, " D D"],
    *,
    tol: float,
) -> tuple[Float[Array, " m" ] | None, Float[Array, " m m"] | None]:
    idx = jnp.asarray(tuple(rows), dtype=int)
    Bsub = B[idx]
    csub = c[idx]
    m = int(Bsub.shape[0])
    # Solve [csub; Bsub^T] beta = [1; 0]
    system = jnp.zeros((m, m))
    system = system.at[0, :].set(csub)
    system = system.at[1:, :].set(Bsub.T)
    rhs = jnp.zeros((m,))
    rhs = rhs.at[0].set(1.0)
    try:
        beta = jnp.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None, None
    if not bool(jnp.all(jnp.isfinite(beta)).item()):
        return None, None
    W = (Bsub @ J) @ Bsub.T
    return beta, W


def _subset_capacity_candidate(
    beta: Float[Array, " m"],
    W: Float[Array, " m m"],
    *,
    tol: float,
) -> float | None:
    m = int(beta.shape[0])
    if m > 7:
        raise NotImplementedError("Permutation search limited to m<=7 in exp1")
    indices = range(m)
    maximal = float("-inf")
    for ordering in itertools.permutations(indices):
        total = 0.0
        for i in range(1, m):
            ii = ordering[i]
            wi = float(beta[ii])
            if wi <= float(tol):
                continue
            row = W[ii]
            for j in range(i):
                jj = ordering[j]
                wj = float(beta[jj])
                if wj <= float(tol):
                    continue
                total += wi * wj * float(row[jj])
        maximal = max(maximal, total)
    if maximal <= float(tol):
        return None
    return 0.5 / maximal
