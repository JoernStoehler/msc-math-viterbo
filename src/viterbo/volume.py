"""Volume estimators for the modern API."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.types import Polytope
from viterbo._wrapped import spatial as _spatial


def volume_reference(bundle: Polytope) -> float:
    """Return a reference volume estimate for ``bundle``.

    - In 2D, compute the exact polygon area via the shoelace formula on the
      convex hull of the provided vertices (ordered by angle around centroid).
    - In higher dimensions, fall back to Qhull volume.
    """
    verts = jnp.asarray(bundle.vertices, dtype=jnp.float64)
    d = int(verts.shape[1])
    if d == 2:
        # Use hull-ordered vertices and the shoelace formula for exactness on simple polytopes.
        order_np = _spatial.convex_hull_vertices(verts)
        P = verts[jnp.asarray(order_np, dtype=jnp.int32)]
        x = P[:, 0]
        y = P[:, 1]
        x_next = jnp.concatenate([x[1:], x[:1]], axis=0)
        y_next = jnp.concatenate([y[1:], y[:1]], axis=0)
        area = 0.5 * jnp.abs(jnp.sum(x * y_next - y * x_next))
        return float(area)
    return float(_spatial.convex_hull_volume(verts))


def volume_padded(
    normals: Float[Array, " batch num_facets dimension"],
    offsets: Float[Array, " batch num_facets"],
    *,
    method: str,
) -> Float[Array, " batch"]:
    """Compute batched volumes using a padding-friendly method.

    Padding semantics:
    - Returns a length-``batch`` vector with in-band invalidation using ``NaN``
      to indicate unbounded/infeasible elements. No separate mask is returned.
    - Future implementations may use SciPy hull volume per batch (JAX-agnostic)
      or a pure-JAX estimator where possible. For now this is a shape-only
      placeholder to keep batching out of the critical path.
    """
    batch = normals.shape[0]
    return jnp.zeros((batch,), dtype=jnp.float64)
