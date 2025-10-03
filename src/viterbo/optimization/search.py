"""Heuristics for exploring candidate counterexamples to Viterbo's conjecture."""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import combinations

import numpy as np

from viterbo.geometry.polytopes import (
    Polytope,
    affine_transform,
    cartesian_product,
    catalog,
    random_affine_map,
    random_polytope,
    regular_polygon_product,
)


def enumerate_search_space(
    *,
    rng_seed: int = 7,
    max_dimension: int = 6,
    transforms_per_base: int = 4,
    random_polytopes_per_dimension: int = 5,
) -> tuple[Polytope, ...]:
    """Return a deterministic tuple of polytopes spanning diverse geometries."""
    rng = np.random.default_rng(rng_seed)
    candidates: list[Polytope] = []

    base_catalog = list(catalog())
    candidates.extend(base_catalog)

    for polytope in base_catalog:
        for index in range(transforms_per_base):
            matrix, translation = random_affine_map(
                polytope.dimension,
                rng=rng,
                translation_scale=0.2,
            )
            transformed = affine_transform(
                polytope,
                matrix,
                translation=translation,
                name=f"{polytope.name}-affine-{index}",
                description=f"Affine image #{index} of {polytope.name}",
            )
            candidates.append(transformed)

    valid_products: list[Polytope] = []
    for first, second in combinations(base_catalog, 2):
        dimension = first.dimension + second.dimension
        if dimension > max_dimension:
            continue
        product_poly = cartesian_product(
            first,
            second,
            name=f"{first.name}-x-{second.name}",
            description=(
                f"Cartesian product spanning dimensions {first.dimension} + {second.dimension}"
            ),
        )
        valid_products.append(product_poly)
    candidates.extend(valid_products)

    if max_dimension >= 4:
        for sides_first in range(5, 9):
            for sides_second in range(5, 9):
                rotation = rng.uniform(0.0, math.pi / 2)
                polygon_product = regular_polygon_product(
                    sides_first,
                    sides_second,
                    rotation=rotation,
                    radius_first=1.0,
                    radius_second=1.0,
                    name=f"{sides_first}gonx{sides_second}gon-{rotation:.2f}",
                )
                candidates.append(polygon_product)

    for dimension in range(2, max_dimension + 1):
        for sample_index in range(random_polytopes_per_dimension):
            random_poly = random_polytope(
                dimension,
                rng=rng,
                name=f"random-{dimension}d-{sample_index}",
            )
            candidates.append(random_poly)

    return tuple(candidates)


def iter_search_space(**kwargs: int) -> Iterator[Polytope]:
    """Yield polytopes from :func:`enumerate_search_space` lazily."""
    for polytope in enumerate_search_space(**kwargs):
        yield polytope
