"""EHZ capacity routines for the modern API.

Implements a JAX-first reference following Haim–Kislev (facet-normal formula)
for even dimensions, and uses the 2D equivalence ``c_EHZ = area``.
No dependencies on legacy modules.
"""

from __future__ import annotations

import itertools
import jax.numpy as jnp
import numpy as _np
from jaxtyping import Array, Float

from viterbo.modern.types import Polytope
from viterbo.modern.volume import volume_reference as _volume_ref


def ehz_capacity_reference(bundle: Polytope) -> float:
    """Return a reference EHZ capacity estimate.

    - In 2D, EHZ capacity coincides with area; reuse the reference volume.
    - In higher even dimensions, compute the facet-normal (Haim–Kislev) value.
    """
    dim = int(bundle.vertices.shape[1])
    if dim == 2:
        return float(_volume_ref(bundle))
    return float(_capacity_reference_facet_normals(bundle.normals, bundle.offsets))


def ehz_capacity_batched(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    max_cycles: int,
) -> Float[Array, " batch"]:
    """Compute batched EHZ capacities with padding.

    Padding semantics:
    - Returns in-band invalidation with ``NaN`` per element that is invalid
      (e.g., infeasible or unbounded halfspace input), without a separate mask.
    - Currently a shape-only placeholder; full batching deferred.
    """
    batch = int(normals.shape[0])
    return jnp.zeros((batch,), dtype=jnp.float64)


def _standard_symplectic_matrix(dimension: int) -> Float[Array, " dim dim"]:
    if dimension % 2 != 0 or dimension < 2:
        raise ValueError("Symplectic matrix requires even dimension >= 2")
    n = dimension // 2
    upper = jnp.hstack((jnp.zeros((n, n)), -jnp.eye(n)))
    lower = jnp.hstack((jnp.eye(n), jnp.zeros((n, n))))
    return jnp.vstack((upper, lower)).astype(jnp.float64)


def _capacity_reference_facet_normals(
    B_matrix: Float[Array, " m d"],
    c_vector: Float[Array, " m"],
    *,
    tol: float = 1e-10,
) -> float:
    """Facet-normal reference (Haim–Kislev) for even dimensions.

    Enumerates all subsets of size ``d+1``, solves Reeb measures via a small
    linear system, and maximizes the quadratic form over permutations to get a
    subset candidate. Returns the minimum over admissible subsets.
    """
    B = jnp.asarray(B_matrix, dtype=jnp.float64)
    c = jnp.asarray(c_vector, dtype=jnp.float64)
    if B.ndim != 2 or c.ndim != 1 or B.shape[0] != c.shape[0]:
        raise ValueError("Invalid half-space system")
    m, d = int(B.shape[0]), int(B.shape[1])
    if d % 2 != 0 or d < 2:
        raise ValueError("Ambient dimension must be even and >= 2")
    subset_size = d + 1
    J = _standard_symplectic_matrix(d)

    best = jnp.inf
    for rows in itertools.combinations(range(m), subset_size):
        idx = jnp.asarray(rows, dtype=jnp.int32)
        Bsub = B[idx]
        csub = c[idx]
        # Solve [csub; Bsub^T] beta = [1; 0] using a tiny Tikhonov
        # regularization to avoid singular systems without exceptions.
        system = jnp.zeros((subset_size, subset_size), dtype=jnp.float64)
        system = system.at[0, :].set(csub)
        system = system.at[1:, :].set(Bsub.T)
        rhs = jnp.zeros((subset_size,), dtype=jnp.float64)
        rhs = rhs.at[0].set(1.0)
        eps = jnp.asarray(1e-12, dtype=jnp.float64)
        beta = jnp.linalg.solve(system + eps * jnp.eye(subset_size, dtype=jnp.float64), rhs)
        if not bool(jnp.all(jnp.isfinite(beta)).item()):
            continue
        W = (Bsub @ J) @ Bsub.T
        cand = _subset_capacity_candidate(beta, W, tol=tol)
        if cand is None:
            continue
        if cand < best:
            best = cand
    if not bool(jnp.isfinite(best)):
        raise ValueError("No admissible facet subset satisfied constraints.")
    return float(best)


def _subset_capacity_candidate(
    beta: Float[Array, " m"],
    W: Float[Array, " m m"],
    *,
    tol: float,
) -> float | None:
    # Convert to NumPy for tiny control-flow heavy loops (m <= 7).
    beta_np = _np.asarray(beta, dtype=float)
    W_np = _np.asarray(W, dtype=float)
    m = int(beta_np.shape[0])
    indices = range(m)
    maximal = float("-inf")
    for ordering in itertools.permutations(indices):
        total = 0.0
        for i in range(1, m):
            ii = ordering[i]
            wi = float(beta_np[ii])
            if wi <= float(tol):
                continue
            row = W_np[ii]
            for j in range(i):
                jj = ordering[j]
                wj = float(beta_np[jj])
                if wj <= float(tol):
                    continue
                total += wi * wj * float(row[jj])
        maximal = max(maximal, total)
    if maximal <= float(tol):
        return None
    return 0.5 / maximal
