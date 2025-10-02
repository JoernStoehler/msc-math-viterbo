"""Canonical polytope families used across tests, docs, and profiling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .ehz import standard_symplectic_matrix

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


@dataclass(frozen=True)
class Polytope:
    """Immutable container describing a convex polytope via half-space data."""

    name: str
    B: FloatMatrix  # shape: (num_facets, dimension)
    c: FloatVector  # shape: (num_facets,)
    description: str = ""
    reference_capacity: float | None = None

    def __post_init__(self) -> None:
        """Normalize inputs and freeze the underlying arrays."""
        matrix = np.asarray(self.B, dtype=float)
        offsets = np.asarray(self.c, dtype=float)

        if matrix.ndim != 2:
            msg = "Facet matrix B must be two-dimensional."
            raise ValueError(msg)

        if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
            msg = "Offsets vector c must match the number of facets."
            raise ValueError(msg)

        object.__setattr__(self, "B", matrix)
        object.__setattr__(self, "c", offsets)

        matrix.setflags(write=False)
        offsets.setflags(write=False)

    @property
    def dimension(self) -> int:
        """Ambient dimension of the polytope."""
        return int(self.B.shape[1])

    @property
    def facets(self) -> int:
        """Number of facet-defining inequalities."""
        return int(self.B.shape[0])

    def halfspace_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of ``(B, c)`` suitable for downstream mutation."""
        return np.array(self.B, copy=True), np.array(self.c, copy=True)


def haim_kislev_action(
    B: np.ndarray,
    c: np.ndarray,
    *,
    subset: Sequence[int],
    order: Sequence[int],
) -> float:
    """Evaluate the Haim–Kislev action for a facet subset and total order."""
    matrix = np.asarray(B, dtype=float)
    offsets = np.asarray(c, dtype=float)
    if matrix.ndim != 2:
        msg = "Facet matrix B must be two-dimensional."
        raise ValueError(msg)

    if offsets.ndim != 1 or offsets.shape[0] != matrix.shape[0]:
        msg = "Vector c must match the number of facets."
        raise ValueError(msg)

    dimension = matrix.shape[1]
    J = standard_symplectic_matrix(dimension)

    rows = np.asarray(tuple(subset), dtype=int)
    B_subset = matrix[rows]
    c_subset = offsets[rows]
    m = len(rows)

    system = np.zeros((m, m))
    system[0, :] = c_subset
    system[1:, :] = B_subset.T

    rhs = np.zeros(m)
    rhs[0] = 1.0
    beta = np.linalg.solve(system, rhs)

    symplectic_products = (B_subset @ J) @ B_subset.T

    total = 0.0
    for i in range(1, m):
        idx_i = order[i]
        weight_i = beta[idx_i]
        if weight_i <= 0.0:
            continue
        row = symplectic_products[idx_i]
        for j in range(i):
            idx_j = order[j]
            weight_j = beta[idx_j]
            if weight_j <= 0.0:
                continue
            total += weight_i * weight_j * row[idx_j]

    if total <= 0.0:
        msg = "Facet ordering yielded a non-positive action."
        raise ValueError(msg)

    return 0.5 / total


def simplex_with_uniform_weights(
    dimension: int,
    *,
    last_offset: float | None = None,
    name: str | None = None,
) -> Polytope:
    """Return the simplex used for regression tests with uniform weights."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

    B = np.eye(dimension)
    extra = -np.ones((1, dimension))
    B = np.vstack((B, extra))

    offsets = np.ones(dimension + 1)
    if last_offset is None:
        last_offset = dimension / 2
    offsets[-1] = float(last_offset)

    polytope_name = name or f"uniform-simplex-{dimension}d"
    reference_capacity = None
    if dimension == 4 and math.isclose(last_offset, dimension / 2):
        reference_capacity = 9.0

    description = (
        "4D simplex with symmetric coordinates; used as a canonical regression example"
        if dimension == 4
        else "Simplex with uniform coordinate facets and a balancing facet."
    )
    return Polytope(
        name=polytope_name,
        B=B,
        c=offsets,
        description=description,
        reference_capacity=reference_capacity,
    )


def truncated_simplex_four_dim() -> Polytope:
    """Return the 4D simplex truncated by an additional slanted facet."""
    B = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    c = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.2])
    reference_capacity = haim_kislev_action(
        B,
        c,
        subset=(0, 1, 2, 3, 4),
        order=(2, 0, 4, 3, 1),
    )
    description = "Simplex-like model with an extra facet; preserves the optimal Reeb action"
    return Polytope(
        name="truncated-simplex-4d",
        B=B,
        c=c,
        description=description,
        reference_capacity=reference_capacity,
    )


def cross_polytope(
    dimension: int,
    *,
    radius: float = 1.0,
    name: str | None = None,
) -> Polytope:
    """Return the centrally symmetric cross-polytope of the given radius."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

    normals = np.array(list(product((-1.0, 1.0), repeat=dimension)))
    c = np.full(normals.shape[0], float(radius))
    description = "Centrally symmetric cross-polytope with L1 ball geometry."
    return Polytope(
        name=name or f"cross-polytope-{dimension}d",
        B=normals,
        c=c,
        description=description,
    )


def hypercube(
    dimension: int,
    *,
    radius: float = 1.0,
    name: str | None = None,
) -> Polytope:
    """Return the axis-aligned hypercube of side length ``2 * radius``."""
    if dimension < 2:
        msg = "Dimension must be at least two."
        raise ValueError(msg)

    identity = np.eye(dimension)
    B = np.vstack((identity, -identity))
    c = np.full(2 * dimension, float(radius))
    description = "Hypercube aligned with the coordinate axes."
    return Polytope(
        name=name or f"hypercube-{dimension}d",
        B=B,
        c=c,
        description=description,
    )


def _regular_polygon_normals(sides: int) -> np.ndarray:
    if sides < 3:
        msg = "A polygon requires at least three sides."
        raise ValueError(msg)

    angles = 2 * np.pi * (np.arange(sides) / sides)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    return normals


def _rotation_matrix(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def regular_polygon_product(
    sides_first: int,
    sides_second: int,
    *,
    rotation: float = 0.0,
    radius_first: float = 1.0,
    radius_second: float | None = None,
    name: str | None = None,
    description: str | None = None,
) -> Polytope:
    r"""Return the direct product of two regular polygons in ``\mathbb{R}^4``."""
    normals_first = _regular_polygon_normals(sides_first)
    normals_second = _regular_polygon_normals(sides_second)
    if radius_second is None:
        radius_second = radius_first

    rotation_matrix = _rotation_matrix(rotation)
    rotated_second = normals_second @ rotation_matrix.T

    zero_block = np.zeros((normals_first.shape[0], 2))
    B_upper = np.hstack((normals_first, zero_block))
    B_lower = np.hstack((np.zeros((rotated_second.shape[0], 2)), rotated_second))
    B = np.vstack((B_upper, B_lower))
    c = np.concatenate(
        (
            np.full(normals_first.shape[0], float(radius_first)),
            np.full(rotated_second.shape[0], float(radius_second)),
        )
    )

    default_name = name or (
        f"{sides_first}gonx{sides_second}gon-rot{int(round(math.degrees(rotation)))}"
    )
    default_description = description or (
        "Product of two regular polygons, yielding a 4D polytope with"
        f" {normals_first.shape[0] + rotated_second.shape[0]} facets."
    )
    return Polytope(
        name=default_name,
        B=B,
        c=c,
        description=default_description,
    )


def viterbo_counterexample(radius: float = 1.0) -> Polytope:
    """Return the Chaidez–Hutchings counterexample to Viterbo's conjecture."""
    description = (
        "Product of a regular pentagon with its quarter-turned copy,"
        " the Chaidez–Hutchings counterexample to Viterbo's conjecture."
    )
    return regular_polygon_product(
        5,
        5,
        rotation=math.pi / 2,
        radius_first=radius,
        radius_second=radius,
        name="viterbo-counterexample",
        description=description,
    )


def random_transformations(
    polytope: Polytope,
    *,
    rng: np.random.Generator,
    count: int,
    scale_range: tuple[float, float] = (0.6, 1.4),
    translation_scale: float = 0.3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate random linear transformations and translations of ``polytope``."""
    lower, upper = scale_range
    if lower <= 0 or upper <= 0:
        msg = "Scaling factors must be positive."
        raise ValueError(msg)

    dimension = polytope.dimension
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(count):
        random_matrix = rng.normal(size=(dimension, dimension))
        q, _ = np.linalg.qr(random_matrix)
        scales = rng.uniform(lower, upper, size=dimension)
        transform = q @ np.diag(scales)
        transformed_B = polytope.B @ transform

        translation = rng.normal(scale=translation_scale, size=dimension)
        transformed_c = polytope.c + transformed_B @ translation

        results.append((transformed_B, transformed_c))
    return results


def catalog() -> tuple[Polytope, ...]:
    """Return a curated tuple of polytopes used for regression and profiling."""
    simplex4 = simplex_with_uniform_weights(4, name="simplex-4d")
    truncated = truncated_simplex_four_dim()
    simplex6 = simplex_with_uniform_weights(6, name="simplex-6d")
    hexagon_product = regular_polygon_product(
        6,
        6,
        rotation=math.pi / 6,
        name="hexagon-product-rot30",
        description="Product of two hexagons; features twelve facets in dimension four.",
    )
    counterexample = viterbo_counterexample()
    return simplex4, truncated, simplex6, hexagon_product, counterexample


__all__ = [
    "Polytope",
    "catalog",
    "cross_polytope",
    "haim_kislev_action",
    "hypercube",
    "random_transformations",
    "regular_polygon_product",
    "simplex_with_uniform_weights",
    "truncated_simplex_four_dim",
    "viterbo_counterexample",
]
