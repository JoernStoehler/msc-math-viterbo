"""Shared standard polytopes used across tests.

The fixtures here provide a small yet diverse catalogue of polytopes with
pre-computed invariants.  Tests can import these instead of hand-rolling
geometry in each module.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import torch

from viterbo.math.capacity_ehz.common import polygon_area
from viterbo.math.constructions import lagrangian_product, rotated_regular_ngon2d
from viterbo.math.polytope import vertices_to_halfspaces
from viterbo.math.volume import volume as volume_from_vertices

DTYPE = torch.float64


@dataclasses.dataclass(frozen=True)
class StandardPolytope:
    """Canonical test polytope with metadata and invariants."""

    name: str
    description: str
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor
    volume: torch.Tensor
    volume_reference: float | None
    capacity_ehz_reference: float | None
    tags: tuple[str, ...] = ()
    references: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return int(self.vertices.size(1))

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.size(0))

    @property
    def num_facets(self) -> int:
        return int(self.normals.size(0))


def _square_axis_aligned() -> StandardPolytope:
    vertices = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=DTYPE,
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    volume = volume_from_vertices(vertices)
    return StandardPolytope(
        name="square_2d",
        description="Axis-aligned unit square; baseline polygon for capacity sanity checks.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        volume_reference=4.0,
        capacity_ehz_reference=4.0,
        tags=("planar", "symmetric"),
        references={"capacity_ehz": "Area equals EHZ capacity for convex polygons in R^2."},
    )


def _random_hexagon_seed41() -> StandardPolytope:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    angles = torch.sort(torch.rand(6, generator=generator) * (2.0 * torch.pi)).values
    radii = 0.8 + 0.4 * torch.rand(6, generator=generator)
    vertices = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1).to(DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    area = polygon_area(vertices)
    reference = float(area.item())
    return StandardPolytope(
        name="random_hexagon_seed41",
        description="Planar random hexagon (seed 41) probing generic polygon combinatorics.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=area,
        volume_reference=reference,
        capacity_ehz_reference=reference,
        tags=("planar", "randomised"),
        references={"capacity_ehz": "Capacity equals area in 2D; computed deterministically."},
    )


def _orthogonal_simplex_4d() -> StandardPolytope:
    origin = torch.zeros((1, 4), dtype=DTYPE)
    basis = torch.eye(4, dtype=DTYPE)
    vertices = torch.cat((origin, basis), dim=0)
    normals, offsets = vertices_to_halfspaces(vertices)
    volume = volume_from_vertices(vertices)
    return StandardPolytope(
        name="orthogonal_simplex_4d",
        description="Orthogonal 4-simplex (origin plus basis vectors); minimal vertex 4D case.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        volume_reference=1.0 / math.factorial(4),
        capacity_ehz_reference=None,
        tags=("4d", "simplicial"),
        references={"volume": "Standard simplex volume 1/4!."},
    )


def _hypercube_4d_unit() -> StandardPolytope:
    corners = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
    )
    vertices = corners.to(DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    volume = volume_from_vertices(vertices)
    return StandardPolytope(
        name="hypercube_4d_unit",
        description="Axis-aligned 4D hypercube [-1, 1]^4 highlighting high symmetry.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        volume_reference=16.0,
        capacity_ehz_reference=None,
        tags=("4d", "symmetric"),
        references={"volume": "Product of edge lengths (2^4)."},
    )


def _pentagon_product_counterexample() -> StandardPolytope:
    vertices_q, _, _ = rotated_regular_ngon2d(5, 0.0)
    vertices_p, _, _ = rotated_regular_ngon2d(5, -0.5 * torch.pi)
    vertices, normals, offsets = lagrangian_product(
        vertices_q.to(DTYPE), vertices_p.to(DTYPE)
    )
    volume = volume_from_vertices(vertices)
    capacity = 2.0 * math.cos(math.pi / 10.0) * (1.0 + math.cos(math.pi / 5.0))
    return StandardPolytope(
        name="pentagon_product_counterexample",
        description=(
            "Regular pentagon × 90° rotation; Haim-Kislev & Ostrover counterexample to Viterbo."
        ),
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        volume_reference=float(volume.item()),
        capacity_ehz_reference=capacity,
        tags=("4d", "symmetric", "lagrangian_product"),
        references={"capacity_ehz": "Haim-Kislev & Ostrover (2024)."},
    )


def _random_polytope_4d_seed2024() -> StandardPolytope:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20241017)
    raw = torch.randn((8, 4), generator=generator, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True)
    directions = raw / norms
    radii = torch.rand((8, 1), generator=generator, dtype=DTYPE) ** (1.0 / 4.0)
    vertices = directions * radii
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    normals, offsets = vertices_to_halfspaces(vertices)
    volume = volume_from_vertices(vertices)
    return StandardPolytope(
        name="random_polytope_4d_seed2024",
        description="Seeded random 4D polytope offering generic combinatorics without huge size.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        volume_reference=float(volume.item()),
        capacity_ehz_reference=None,
        tags=("4d", "randomised"),
        references={},
    )


STANDARD_POLYTOPES: Final[tuple[StandardPolytope, ...]] = (
    _square_axis_aligned(),
    _random_hexagon_seed41(),
    _orthogonal_simplex_4d(),
    _hypercube_4d_unit(),
    _pentagon_product_counterexample(),
    _random_polytope_4d_seed2024(),
)

STANDARD_POLYTOPES_BY_NAME: Final[dict[str, StandardPolytope]] = {
    polytope.name: polytope for polytope in STANDARD_POLYTOPES
}

