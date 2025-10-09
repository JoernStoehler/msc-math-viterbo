from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from viterbo._wrapped.spatial import convex_hull_equations


def enumerate_vertices(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    atol: float = 1e-9,
) -> Float[Array, " k dim"]:
    """Enumerate vertices of the polytope {x : Bx ≤ c} by facet intersections.

    Notes:
      - Solves all d×d systems from facet combinations; filters feasible points.
      - Deduplicates vertices using rounding by ``atol``.
    """
    B = jnp.asarray(normals, dtype=jnp.float64)
    c = jnp.asarray(offsets, dtype=jnp.float64)
    m, dim = int(B.shape[0]), int(B.shape[1])
    if m < dim:
        return jnp.zeros((0, dim), dtype=jnp.float64)

    vertices: list[jnp.ndarray] = []
    for idx in itertools.combinations(range(m), dim):
        rows = jnp.asarray(idx, dtype=int)
        M = B[rows, :]
        rhs = c[rows]
        try:
            x = jnp.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        if bool(jnp.all(B @ x <= c + float(atol)).item()):
            vertices.append(jnp.asarray(x, dtype=jnp.float64))
    if not vertices:
        return jnp.zeros((0, dim), dtype=jnp.float64)

    verts = jnp.stack(vertices, axis=0)
    # Deduplicate by rounding w.r.t. atol
    keys = jnp.round(verts / float(atol)).astype(int)
    # Preserve order deterministically
    seen: set[tuple[int, ...]] = set()
    unique: list[jnp.ndarray] = []
    for i in range(int(keys.shape[0])):
        tup = tuple(int(x) for x in keys[i].tolist())
        if tup in seen:
            continue
        seen.add(tup)
        unique.append(verts[i])
    return jnp.stack(unique, axis=0) if unique else jnp.zeros((0, dim), dtype=jnp.float64)


def halfspaces_from_vertices(
    verts: Float[Array, " k dim"],
) -> tuple[Float[Array, " m dim"], Float[Array, " m"]]:
    """Compute a half-space system Bx ≤ c for the convex hull of ``V`` using Qhull."""
    verts = jnp.asarray(verts, dtype=jnp.float64)
    eq = convex_hull_equations(verts)
    normals = eq[:, :-1]
    offsets = eq[:, -1]
    B = jnp.asarray(normals, dtype=jnp.float64)
    c = jnp.asarray(-offsets, dtype=jnp.float64)
    return B, c


@dataclass(frozen=True)
class HalfspaceDegeneracyReport:
    """Summary metrics indicating (near-)degeneracy of a half-space system.

    Notes:
      - This is a diagnostic helper for development and debugging in exp1.
      - Library algorithms assume inputs are already validated elsewhere; callers
        can use this function opportunistically to inspect conditioning.
    """

    m: int
    dim: int
    rank: int
    min_singular_value: Float[Array, ""]
    condition_number: Float[Array, ""]
    max_abs_row_correlation: Float[Array, ""]
    duplicate_facet_fraction: Float[Array, ""]
    vertex_count: Optional[int]
    simple_vertex_count: Optional[int]
    min_simple_vertex_sigma: Optional[Float[Array, ""]]
    max_simple_vertex_condition: Optional[Float[Array, ""]]


def halfspace_degeneracy_metrics(
    normals: Float[Array, " m dim"],
    offsets: Float[Array, " m"],
    *,
    atol: float = 1e-9,
    with_vertices: bool = False,
) -> HalfspaceDegeneracyReport:
    """Compute light-weight degeneracy indicators for ``Bx ≤ c``.

    Returns a report with matrix-level metrics (always) and optional vertex-level
    metrics when ``with_vertices=True``. The function is pure and does not modify
    inputs. It performs no validation beyond basic shape usage.
    """
    B = jnp.asarray(normals, dtype=jnp.float64)
    c = jnp.asarray(offsets, dtype=jnp.float64)
    m = int(B.shape[0])
    dim = int(B.shape[1]) if B.ndim == 2 else 0

    # SVD-based conditioning of the facet matrix (as a linear map).
    if m == 0 or dim == 0:
        s_min = jnp.asarray(0.0, dtype=jnp.float64)
        cond = jnp.asarray(jnp.inf, dtype=jnp.float64)
        rank = 0
    else:
        s = jnp.linalg.svd(B, full_matrices=False, compute_uv=False)
        s_min = jnp.min(s)
        s_max = jnp.max(s)
        # Guard the denominator to avoid NaNs; interpret as ill-conditioned.
        cond = s_max / jnp.clip(s_min, a_min=jnp.asarray(atol, dtype=jnp.float64), a_max=None)
        rank = int(jnp.sum(s > float(atol)))

    # Row-direction correlations (detect nearly parallel facets).
    if m <= 1:
        max_corr = jnp.asarray(0.0, dtype=jnp.float64)
        dup_frac = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        norms = jnp.linalg.norm(B, axis=1, keepdims=True)
        norms = jnp.clip(norms, a_min=1e-12)
        rows = B / norms
        gram = rows @ rows.T
        gram = gram - jnp.eye(m)
        max_corr = jnp.max(jnp.abs(gram))
        # Duplicate rows (directional duplicates) via rounding keys.
        keys = jnp.round(rows / float(atol)).astype(int)
        seen: set[tuple[int, ...]] = set()
        unique = 0
        for i in range(m):
            tup = tuple(int(x) for x in keys[i].tolist())
            if tup in seen:
                continue
            seen.add(tup)
            unique += 1
        dup_frac = jnp.asarray(0.0 if m == 0 else 1.0 - (unique / float(m)), dtype=jnp.float64)

    v_count: Optional[int] = None
    simple_count: Optional[int] = None
    v_smin: Optional[Float[Array, ""]] = None
    v_cond: Optional[Float[Array, ""]] = None

    if with_vertices and dim > 0 and m >= dim:
        from viterbo.exp1.halfspaces import enumerate_vertices as _enumerate

        V = _enumerate(B, c, atol=atol)
        v_count = int(V.shape[0])
        smins: list[float] = []
        conds: list[float] = []
        simple_count = 0
        for k in range(v_count):
            x = V[k]
            residuals = B @ x - c
            active = jnp.where(jnp.abs(residuals) <= float(atol))[0]
            if int(active.shape[0]) != dim:
                continue
            simple_count += 1
            A_act = B[active, :]
            s_loc = jnp.linalg.svd(A_act, full_matrices=False, compute_uv=False)
            smin = float(jnp.min(s_loc))
            smax = float(jnp.max(s_loc))
            smins.append(smin)
            conds.append(smax / max(smin, float(atol)))
        if smins:
            v_smin = jnp.asarray(min(smins), dtype=jnp.float64)
            v_cond = jnp.asarray(max(conds), dtype=jnp.float64)

    return HalfspaceDegeneracyReport(
        m=m,
        dim=dim,
        rank=rank,
        min_singular_value=jnp.asarray(s_min, dtype=jnp.float64),
        condition_number=jnp.asarray(cond, dtype=jnp.float64),
        max_abs_row_correlation=jnp.asarray(max_corr, dtype=jnp.float64),
        duplicate_facet_fraction=jnp.asarray(dup_frac, dtype=jnp.float64),
        vertex_count=v_count,
        simple_vertex_count=simple_count,
        min_simple_vertex_sigma=v_smin,
        max_simple_vertex_condition=v_cond,
    )

