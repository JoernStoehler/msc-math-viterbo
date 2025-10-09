from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.exp1.capacity_ehz import capacity
from viterbo.exp1.polytopes import HalfspacePolytope, LagrangianProductPolytope, Polytope
from viterbo.exp1.volume import volume


def systolic_ratio(P: Polytope, *, method: str = "auto") -> Float[Array, ""]:
    """Return sys(K) = c(K)^n / (n! vol_{2n}(K)) for even-dim polytopes.

    Notes:
      - Uses exp1.capacity_ehz.capacity and exp1.volume.volume.
      - "auto" dispatch mirrors capacity().
    """
    if isinstance(P, HalfspacePolytope):
        dim = int(P.normals.shape[1])
    elif isinstance(P, LagrangianProductPolytope):
        dim = 4
    else:
        dim = 4
    if dim % 2 != 0:
        raise ValueError("Systolic ratio defined for even dimensions only.")
    n = dim // 2
    cap = float(capacity(P, method=method))
    vol = float(volume(P, method="fast"))
    denom = math.factorial(n) * vol
    if denom <= 0.0:
        raise ValueError("Volume must be positive.")
    return jnp.asarray((cap**n) / denom, dtype=jnp.float64)
