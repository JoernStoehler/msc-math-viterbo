"""Dataset-facing cycle helpers built on the oriented-edge graph."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from viterbo.datasets.types import Polytope
from viterbo.math.capacity.reeb_cycles import minimum_cycle_reference as _minimum_cycle_reference
from viterbo.math.numerics import GEOMETRY_ABS_TOLERANCE


def minimum_cycle_reference(
    polytope: Polytope,
    *,
    atol: float = GEOMETRY_ABS_TOLERANCE,
) -> Float[Array, " num_points dimension"]:
    """Return a representative minimum-action cycle for ``polytope``."""

    normals, offsets = polytope.halfspace_data()
    cycle = _minimum_cycle_reference(normals, offsets, atol=atol)
    if cycle.ndim != 2:
        dimension = int(polytope.dimension)
        return jnp.zeros((0, dimension), dtype=jnp.float64)
    return cycle


__all__ = ["minimum_cycle_reference"]
