"""Heuristics for exploring candidate counterexamples to Viterbo's conjecture."""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import combinations, islice

import jax
import jax.numpy as jnp

from viterbo.geometry.polytopes import (
    Polytope,
    affine_transform,
    cartesian_product,
    catalog,
    random_affine_map,
    random_polytope,
    regular_polygon_product,
)


def _generate_search_space(
    *,
    rng_seed: int,
    max_dimension: int,
    transforms_per_base: int,
    random_polytopes_per_dimension: int,
) -> Iterator[Polytope]:
    key = jax.random.PRNGKey(rng_seed)
    base_catalog = tuple(catalog())

    yield from base_catalog

    for polytope in base_catalog:
        for index in range(transforms_per_base):
            key, subkey = jax.random.split(key)
            matrix, translation = random_affine_map(
                polytope.dimension,
                key=subkey,
                translation_scale=0.2,
            )
            matrix_inv = jnp.linalg.inv(jnp.asarray(matrix))
            yield affine_transform(
                polytope,
                matrix,
                translation=translation,
                matrix_inverse=matrix_inv,
                name=f"{polytope.name}-affine-{index}",
                description=f"Affine image #{index} of {polytope.name}",
            )

    for first, second in combinations(base_catalog, 2):
        dimension = first.dimension + second.dimension
        if dimension > max_dimension:
            continue
        yield cartesian_product(
            first,
            second,
            name=f"{first.name}-x-{second.name}",
            description=(
                f"Cartesian product spanning dimensions {first.dimension} + {second.dimension}"
            ),
        )

    if max_dimension >= 4:
        for sides_first in range(5, 9):
            for sides_second in range(5, 9):
                key, rkey = jax.random.split(key)
                rotation = float(jax.random.uniform(rkey, (), minval=0.0, maxval=math.pi / 2))
                yield regular_polygon_product(
                    sides_first,
                    sides_second,
                    rotation=rotation,
                    radius_first=1.0,
                    radius_second=1.0,
                    name=f"{sides_first}gonx{sides_second}gon-{rotation:.2f}",
                )

    for dimension in range(2, max_dimension + 1):
        for sample_index in range(random_polytopes_per_dimension):
            key, pkey = jax.random.split(key)
            yield random_polytope(
                dimension,
                key=pkey,
                name=f"random-{dimension}d-{sample_index}",
            )


def enumerate_search_space(
    *,
    rng_seed: int = 7,
    max_dimension: int = 6,
    transforms_per_base: int = 4,
    random_polytopes_per_dimension: int = 5,
    max_candidates: int | None = None,
) -> tuple[Polytope, ...]:
    """Return a deterministic tuple of polytopes spanning diverse geometries.

    Args:
      rng_seed: Seed controlling the pseudo-random search components.
      max_dimension: Maximum ambient dimension for enumerated polytopes.
      transforms_per_base: Number of affine transforms applied to each base polytope.
      random_polytopes_per_dimension: Random samples generated per dimension.
      max_candidates: Optional cap on the number of polytopes produced.
    """
    generator = _generate_search_space(
        rng_seed=rng_seed,
        max_dimension=max_dimension,
        transforms_per_base=transforms_per_base,
        random_polytopes_per_dimension=random_polytopes_per_dimension,
    )
    if max_candidates is None:
        return tuple(generator)
    return tuple(islice(generator, max_candidates))


def iter_search_space(**kwargs: int) -> Iterator[Polytope]:
    """Yield polytopes from :func:`enumerate_search_space` lazily."""
    max_candidates = kwargs.pop("max_candidates", None)
    rng_seed = kwargs.pop("rng_seed", 7)
    max_dimension = kwargs.pop("max_dimension", 6)
    transforms_per_base = kwargs.pop("transforms_per_base", 4)
    random_polytopes_per_dimension = kwargs.pop("random_polytopes_per_dimension", 5)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        msg = f"Unknown keyword arguments for iter_search_space: {unexpected}"
        raise TypeError(msg)

    generator = _generate_search_space(
        rng_seed=rng_seed,
        max_dimension=max_dimension,
        transforms_per_base=transforms_per_base,
        random_polytopes_per_dimension=random_polytopes_per_dimension,
    )
    if max_candidates is None:
        yield from generator
    else:
        yield from islice(generator, max_candidates)
