"""Systolic ratio checks on canonical 4D shapes."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from viterbo.datasets2 import generators
from viterbo.math.capacity.facet_normals import (
    ehz_capacity_reference_facet_normals as cap_ref,
)
from viterbo.math.volume import polytope_volume_reference as vol_ref


@pytest.mark.goal_math
@pytest.mark.smoke
def test_systolic_ratio_simplex_4d_equals_three_quarters() -> None:
    """Right 4D simplex conv(0,2e_i) has systolic ratio 3/4 exactly."""
    simp = generators.simplex(4)
    cap = cap_ref(simp.normals, simp.offsets)
    vol = vol_ref(simp.normals, simp.offsets)
    sys = (cap**2) / (math.factorial(2) * vol)
    assert jnp.isclose(sys, 0.75, rtol=1e-10, atol=0.0)


@pytest.mark.goal_math
@pytest.mark.deep
def test_systolic_ratio_less_than_one_for_hypercube_and_cross_polytope() -> None:
    """Hypercube and cross-polytope 4D systolic ratios are below the unit-ball bound."""
    cube = generators.hypercube(4, radius=1.0)
    cross = generators.cross_polytope(4, radius=1.0)
    cap_cube = cap_ref(cube.normals, cube.offsets)
    cap_cross = cap_ref(cross.normals, cross.offsets)
    vol_cube = vol_ref(cube.normals, cube.offsets)
    vol_cross = vol_ref(cross.normals, cross.offsets)
    sys_cube = (cap_cube**2) / (math.factorial(2) * vol_cube)
    sys_cross = (cap_cross**2) / (math.factorial(2) * vol_cross)
    assert sys_cube < 1.0
    assert sys_cross < 1.0
