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


def _segment_1d_neg2_3p5() -> StandardPolytope:
    vertices = torch.tensor([[-2.0], [3.5]], dtype=DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    length = volume_from_vertices(vertices)
    return StandardPolytope(
        name="segment_1d_neg2_3p5",
        description="1D segment spanning [-2, 3.5]; used for 1D volume sanity checks.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=length,
        volume_reference=5.5,
        capacity_ehz_reference=None,
        tags=("1d",),
    )


def _segment_1d_symmetric_unit() -> StandardPolytope:
    vertices = torch.tensor([[-1.0], [1.0]], dtype=DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    length = volume_from_vertices(vertices)
    return StandardPolytope(
        name="segment_1d_symmetric_unit",
        description="1D symmetric unit segment [-1, 1] used in product constructions.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=length,
        volume_reference=2.0,
        capacity_ehz_reference=None,
        tags=("1d", "symmetric"),
    )


def _segment_1d_shifted_length2() -> StandardPolytope:
    vertices = torch.tensor([[0.0], [2.0]], dtype=DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    length = volume_from_vertices(vertices)
    return StandardPolytope(
        name="segment_1d_shifted_length2",
        description="1D segment [0, 2] used for Lagrangian product smoke tests.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=length,
        volume_reference=2.0,
        capacity_ehz_reference=None,
        tags=("1d",),
    )


def _right_triangle_area_one() -> StandardPolytope:
    vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    area = volume_from_vertices(vertices)
    return StandardPolytope(
        name="right_triangle_area_one",
        description="Right triangle with area one for 2D volume checks.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=area,
        volume_reference=1.0,
        capacity_ehz_reference=None,
        tags=("planar",),
    )


def _tetrahedron_box_123() -> StandardPolytope:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=DTYPE,
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    vol = volume_from_vertices(vertices)
    return StandardPolytope(
        name="tetrahedron_box_123",
        description="Axis-aligned tetrahedron with edges (1, 2, 3) yielding volume 1.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=vol,
        volume_reference=1.0,
        capacity_ehz_reference=None,
        tags=("3d", "simplicial"),
    )


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


def _hypercube_3d_unit() -> StandardPolytope:
    corners = torch.cartesian_prod(
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
        torch.tensor([-1.0, 1.0], dtype=DTYPE),
    )
    vertices = corners.to(DTYPE)
    normals, offsets = vertices_to_halfspaces(vertices)
    volume = volume_from_vertices(vertices)
    return StandardPolytope(
        name="hypercube_3d_unit",
        description="Axis-aligned 3D cube [-1, 1]^3 used in volume sanity checks.",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        volume=volume,
        volume_reference=8.0,
        capacity_ehz_reference=None,
        tags=("3d", "symmetric"),
        references={"volume": "Product of edge lengths (2^3)."},
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
    _segment_1d_neg2_3p5(),
    _segment_1d_symmetric_unit(),
    _segment_1d_shifted_length2(),
    _right_triangle_area_one(),
    _tetrahedron_box_123(),
    _square_axis_aligned(),
    _random_hexagon_seed41(),
    _orthogonal_simplex_4d(),
    _hypercube_3d_unit(),
    _hypercube_4d_unit(),
    _pentagon_product_counterexample(),
    _random_polytope_4d_seed2024(),
)

STANDARD_POLYTOPES_BY_NAME: Final[dict[str, StandardPolytope]] = {
    polytope.name: polytope for polytope in STANDARD_POLYTOPES
}


@dataclasses.dataclass(frozen=True)
class PlanarPolytope:
    """Planar polygon with cached H-representation and area."""

    name: str
    description: str
    vertices: torch.Tensor
    normals: torch.Tensor
    offsets: torch.Tensor
    area: torch.Tensor
    tags: tuple[str, ...] = ()
    references: dict[str, str] = dataclasses.field(default_factory=dict)


def _regular_pentagon() -> PlanarPolytope:
    vertices, normals, offsets = rotated_regular_ngon2d(5, 0.0)
    area = polygon_area(vertices)
    return PlanarPolytope(
        name="regular_pentagon",
        description="Regular pentagon in the symplectic q-plane (unit radius).",
        vertices=vertices.to(dtype=DTYPE),
        normals=normals.to(dtype=DTYPE),
        offsets=offsets.to(dtype=DTYPE),
        area=area.to(dtype=DTYPE),
        tags=("planar", "symmetric"),
    )


def _rotated_pentagon_90deg() -> PlanarPolytope:
    vertices, normals, offsets = rotated_regular_ngon2d(5, -0.5 * torch.pi)
    area = polygon_area(vertices)
    return PlanarPolytope(
        name="rotated_pentagon_90deg",
        description="Regular pentagon rotated by 90 degrees for Lagrangian product benchmarks.",
        vertices=vertices.to(dtype=DTYPE),
        normals=normals.to(dtype=DTYPE),
        offsets=offsets.to(dtype=DTYPE),
        area=area.to(dtype=DTYPE),
        tags=("planar", "symmetric"),
    )


def _square_planar() -> PlanarPolytope:
    square = STANDARD_POLYTOPES_BY_NAME["square_2d"]
    return PlanarPolytope(
        name="square_planar",
        description="Planar square reused for Lagrangian product smoke tests.",
        vertices=square.vertices,
        normals=square.normals,
        offsets=square.offsets,
        area=square.volume,
        tags=("planar", "symmetric"),
    )


def _minkowski_three_bounce_q() -> PlanarPolytope:
    vertices = torch.tensor(
        [
            [-1.70172287, -1.26811867],
            [-1.56413513, -1.89508182],
            [1.19150140, -0.84596686],
            [1.06428033, 0.66886455],
            [-1.40051732, -0.49548974],
        ],
        dtype=DTYPE,
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    area = polygon_area(vertices)
    return PlanarPolytope(
        name="minkowski_three_bounce_q",
        description="Five-vertex polygon driving the three-bounce orbit Minkowski billiard test (q-plane).",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        area=area,
        tags=("planar", "randomised"),
    )


def _minkowski_three_bounce_p() -> PlanarPolytope:
    vertices = torch.tensor(
        [
            [-1.92487754, -0.41382604],
            [1.65055805, 0.15091143],
            [0.53522767, 0.94426645],
            [1.30412803, 1.78716638],
            [-1.37802231, 1.88839061],
        ],
        dtype=DTYPE,
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    area = polygon_area(vertices)
    return PlanarPolytope(
        name="minkowski_three_bounce_p",
        description="Companion polygon for the three-bounce Minkowski billiard benchmark (p-plane).",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        area=area,
        tags=("planar", "randomised"),
    )


def _minkowski_invariance_q() -> PlanarPolytope:
    vertices = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.2],
            [1.5, 1.5],
            [-0.5, 1.3],
            [-1.2, 0.2],
        ],
        dtype=DTYPE,
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    area = polygon_area(vertices)
    return PlanarPolytope(
        name="minkowski_invariance_q",
        description="Planar polygon used for permutation/translation invariance checks (q-plane).",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        area=area,
        tags=("planar", "randomised"),
    )


def _minkowski_invariance_p() -> PlanarPolytope:
    vertices = torch.tensor(
        [
            [1.2, 0.0],
            [0.3, 1.5],
            [-1.1, 0.7],
            [-0.6, -0.9],
            [1.1, -0.4],
        ],
        dtype=DTYPE,
    )
    normals, offsets = vertices_to_halfspaces(vertices)
    area = polygon_area(vertices)
    return PlanarPolytope(
        name="minkowski_invariance_p",
        description="Companion polygon for permutation/translation invariance checks (p-plane).",
        vertices=vertices,
        normals=normals,
        offsets=offsets,
        area=area,
        tags=("planar", "randomised"),
    )


PLANAR_POLYTOPES: Final[tuple[PlanarPolytope, ...]] = (
    _square_planar(),
    _regular_pentagon(),
    _rotated_pentagon_90deg(),
    _minkowski_three_bounce_q(),
    _minkowski_three_bounce_p(),
    _minkowski_invariance_q(),
    _minkowski_invariance_p(),
)

PLANAR_POLYTOPES_BY_NAME: Final[dict[str, PlanarPolytope]] = {
    polytope.name: polytope for polytope in PLANAR_POLYTOPES
}

PLANAR_POLYTOPE_PAIRS: Final[dict[str, tuple[PlanarPolytope, PlanarPolytope]]] = {
    "square_product": (
        PLANAR_POLYTOPES_BY_NAME["square_planar"],
        PLANAR_POLYTOPES_BY_NAME["square_planar"],
    ),
    "pentagon_product": (
        PLANAR_POLYTOPES_BY_NAME["regular_pentagon"],
        PLANAR_POLYTOPES_BY_NAME["rotated_pentagon_90deg"],
    ),
    "minkowski_three_bounce": (
        PLANAR_POLYTOPES_BY_NAME["minkowski_three_bounce_q"],
        PLANAR_POLYTOPES_BY_NAME["minkowski_three_bounce_p"],
    ),
    "minkowski_invariance": (
        PLANAR_POLYTOPES_BY_NAME["minkowski_invariance_q"],
        PLANAR_POLYTOPES_BY_NAME["minkowski_invariance_p"],
    ),
}
