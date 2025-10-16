from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest
import torch

from viterbo.math.capacity_ehz.algorithms import (
    capacity_ehz_algorithm1,
    capacity_ehz_algorithm2,
    capacity_ehz_primal_dual,
)
from viterbo.math.capacity_ehz.common import polygon_area
from viterbo.math.capacity_ehz.stubs import capacity_ehz_haim_kislev
from viterbo.math.polytope import vertices_to_halfspaces

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class PolytopeCase:
    name: str
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor
    expected: torch.Tensor | None = None
    atol: float = 1e-6
    rtol: float = 1e-6


def _regular_polygon_vertices(sides: int) -> torch.Tensor:
    angles = torch.arange(sides, dtype=torch.get_default_dtype()) * (2.0 * torch.pi / sides)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)


def _l2_cartesian_product(vertices_q: torch.Tensor, vertices_p: torch.Tensor) -> torch.Tensor:
    pairs: list[torch.Tensor] = []
    for q in vertices_q:
        for p in vertices_p:
            pairs.append(torch.cat((q, p)))
    return torch.stack(pairs, dim=0)


def _polygon_case(name: str, vertices: torch.Tensor, expected_value: float | None = None) -> PolytopeCase:
    normals, offsets = vertices_to_halfspaces(vertices)
    expected = None if expected_value is None else torch.tensor(expected_value, dtype=vertices.dtype)
    return PolytopeCase(name=name, vertices=vertices, normals=normals, offsets=offsets, expected=expected)


def _pentagon_product_case() -> PolytopeCase:
    vertices_q = _regular_polygon_vertices(5)
    rotation = -0.5 * torch.pi
    angles = torch.arange(5, dtype=torch.get_default_dtype()) * (2.0 * torch.pi / 5) + rotation
    vertices_p = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    vertices = _l2_cartesian_product(vertices_q, vertices_p)
    normals_q, offsets_q = vertices_to_halfspaces(vertices_q)
    normals_p, offsets_p = vertices_to_halfspaces(vertices_p)
    normals = torch.cat(
        (
            torch.cat((normals_q, torch.zeros(normals_q.size(0), 2, dtype=normals_q.dtype)), dim=1),
            torch.cat((torch.zeros(normals_p.size(0), 2, dtype=normals_p.dtype), normals_p), dim=1),
        ),
        dim=0,
    )
    offsets = torch.cat((offsets_q, offsets_p))
    angle1 = torch.full((), torch.pi / 10.0, dtype=vertices.dtype, device=vertices.device)
    angle2 = torch.full((), torch.pi / 5.0, dtype=vertices.dtype, device=vertices.device)
    expected = 2.0 * torch.cos(angle1) * (1.0 + torch.cos(angle2))
    return PolytopeCase(
        name="pentagon_product",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        expected=expected,
    )


def _all_polytope_cases() -> list[PolytopeCase]:
    torch.manual_seed(41)
    random_angles = torch.sort(torch.rand(6) * (2.0 * torch.pi)).values
    random_radii = 0.8 + 0.4 * torch.rand(6)
    random_vertices = torch.stack((random_radii * torch.cos(random_angles), random_radii * torch.sin(random_angles)), dim=1)
    random_expected = polygon_area(random_vertices)
    cube_vertices = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=4)), dtype=torch.get_default_dtype())
    cube_normals = torch.cat((torch.eye(4), -torch.eye(4)))
    cube_offsets = torch.ones(8, dtype=torch.get_default_dtype())

    return [
        _polygon_case(
            "square",
            torch.tensor(
                [
                    [-1.0, -1.0],
                    [-1.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                ],
                dtype=torch.get_default_dtype(),
            ),
            expected_value=4.0,
        ),
        _polygon_case("random_hexagon", random_vertices, expected_value=float(random_expected.item())),
        _pentagon_product_case(),
        PolytopeCase(
            name="cube4",
            vertices=cube_vertices,
            normals=cube_normals,
            offsets=cube_offsets,
            expected=torch.tensor(4.0, dtype=torch.get_default_dtype()),
        ),
    ]


@pytest.mark.parametrize("case", _all_polytope_cases(), ids=lambda c: c.name)
def test_capacity_algorithms_consistency(case: PolytopeCase) -> None:
    capacity_a1 = capacity_ehz_algorithm1(case.normals, case.offsets)
    capacity_a2 = capacity_ehz_algorithm2(case.vertices)
    capacity_pd = capacity_ehz_primal_dual(case.vertices, case.normals, case.offsets)
    capacity_hk = capacity_ehz_haim_kislev(case.normals, case.offsets)

    torch.testing.assert_close(capacity_a1, capacity_a2, atol=case.atol, rtol=case.rtol)
    torch.testing.assert_close(capacity_pd, capacity_a2, atol=case.atol, rtol=case.rtol)
    torch.testing.assert_close(capacity_hk, capacity_a2, atol=case.atol, rtol=case.rtol)

    if case.expected is not None:
        torch.testing.assert_close(capacity_a2, case.expected, atol=case.atol, rtol=case.rtol)


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
