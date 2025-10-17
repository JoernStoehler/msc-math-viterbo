from __future__ import annotations

import pytest
import torch

from viterbo.math.capacity_ehz.algorithms import (
    capacity_ehz_algorithm1,
    capacity_ehz_algorithm2,
    capacity_ehz_primal_dual,
)
from viterbo.math.capacity_ehz.stubs import capacity_ehz_haim_kislev
from tests.polytopes import STANDARD_POLYTOPES_BY_NAME, StandardPolytope

torch.set_default_dtype(torch.float64)


CAPACITY_TEST_CASES: list[tuple[str, float, float]] = [
    ("square_2d", 1e-6, 1e-6),
    ("random_hexagon_seed41", 1e-6, 1e-6),
    ("hypercube_4d_unit", 1e-5, 1e-5),
    ("pentagon_product_counterexample", 1e-6, 1e-6),
]


@pytest.mark.parametrize("name, atol, rtol", CAPACITY_TEST_CASES)
def test_capacity_algorithms_consistency(name: str, atol: float, rtol: float) -> None:
    polytope: StandardPolytope = STANDARD_POLYTOPES_BY_NAME[name]
    capacity_a1 = capacity_ehz_algorithm1(polytope.normals, polytope.offsets)
    capacity_a2 = capacity_ehz_algorithm2(polytope.vertices)
    capacity_pd = capacity_ehz_primal_dual(polytope.vertices, polytope.normals, polytope.offsets)
    capacity_hk = capacity_ehz_haim_kislev(polytope.normals, polytope.offsets)

    torch.testing.assert_close(capacity_a1, capacity_a2, atol=atol, rtol=rtol)
    torch.testing.assert_close(capacity_pd, capacity_a2, atol=atol, rtol=rtol)
    torch.testing.assert_close(capacity_hk, capacity_a2, atol=atol, rtol=rtol)

    if polytope.capacity_ehz_reference is not None:
        expected = torch.tensor(polytope.capacity_ehz_reference, dtype=capacity_a2.dtype)
        torch.testing.assert_close(capacity_a2, expected, atol=atol, rtol=rtol)


def test_capacity_haim_kislev_rejects_odd_dimension() -> None:
    normals = torch.eye(3)
    offsets = torch.ones(3)
    with pytest.raises(ValueError, match="ambient dimension must be even"):
        capacity_ehz_haim_kislev(normals, offsets)


def test_capacity_haim_kislev_requires_enough_facets() -> None:
    normals = torch.eye(4)
    offsets = torch.ones(4)
    with pytest.raises(ValueError, match="need at least d \\+ 1 facets"):
        capacity_ehz_haim_kislev(normals, offsets)
