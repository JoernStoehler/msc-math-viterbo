"""Profiling helpers for the polytope EHZ capacity implementations."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from viterbo.datasets.catalog import catalog, random_transformations
from viterbo.math.capacity.facet_normals import (
    ehz_capacity_fast_facet_normals,
    ehz_capacity_reference_facet_normals,
)
from viterbo.datasets.types import Polytope as ModernPolytope, PolytopeRecord

Algorithm = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class ProfileConfig:
    """Configuration bundle describing a profiling experiment."""

    label: str
    function: Algorithm
    repeats: int


def _bundle_from_halfspaces(B: np.ndarray, c: np.ndarray) -> ModernPolytope:
    normals = jnp.asarray(B, dtype=jnp.float64)
    offsets = jnp.asarray(c, dtype=jnp.float64)
    dimension = normals.shape[1] if normals.ndim == 2 else 0
    vertices = jnp.empty((0, dimension), dtype=jnp.float64)
    incidence = jnp.empty((0, normals.shape[0]), dtype=bool)
    return ModernPolytope(normals=normals, offsets=offsets, vertices=vertices, incidence=incidence)


ALGORITHMS: dict[str, Algorithm] = {
    "reference": lambda B, c: float(
        ehz_capacity_reference_facet_normals(jnp.asarray(B), jnp.asarray(c))
    ),
    "fast": lambda B, c: float(ehz_capacity_fast_facet_normals(jnp.asarray(B), jnp.asarray(c))),
}


def _registry() -> dict[str, PolytopeRecord]:
    return {entry.metadata.slug: entry for entry in catalog()}


def _parse_args() -> argparse.Namespace:
    registry = _registry()
    parser = argparse.ArgumentParser(
        description="Profile EHZ capacity implementations across canonical polytopes.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS.keys()),
        default=tuple(ALGORITHMS.keys()),
        help="Algorithms to profile (default: all).",
    )
    parser.add_argument(
        "--polytopes",
        nargs="+",
        choices=tuple(registry.keys()),
        default=tuple(registry.keys()),
        help="Subset of named polytopes from the catalog (default: all).",
    )
    parser.add_argument(
        "--transforms",
        type=int,
        default=10,
        help="Number of random affine transforms per base polytope.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for generating affine variants.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions for each algorithm over the dataset.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of rows to display from the profiling statistics.",
    )
    return parser.parse_args()


def _build_dataset(
    polytopes: Iterable[PolytopeRecord],
    *,
    transforms: int,
    seed: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if transforms < 0:
        msg = "Number of transforms must be non-negative."
        raise ValueError(msg)

    key = jax.random.PRNGKey(seed)
    dataset: list[tuple[str, np.ndarray, np.ndarray]] = []
    for poly in polytopes:
        geometry = poly.geometry
        B, c = geometry.halfspace_data()
        dataset.append((poly.metadata.slug, B, c))
        if transforms:
            variants = random_transformations(poly, key=key, count=transforms)
            for index, variant in enumerate(variants):
                variant_B, variant_c = variant.geometry.halfspace_data()
                label = f"{poly.metadata.slug}-variant-{index}"
                dataset.append((label, variant_B, variant_c))
    return dataset


def _profile(
    config: ProfileConfig,
    dataset: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    top: int,
) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(config.repeats):
        for _, B, c in dataset:
            config.function(B, c)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    sys.stdout.write(f"\n=== Profile: {config.label} ===\n")
    stats.print_stats(top)


def main() -> None:
    """Profile the reference and optimized capacity implementations."""
    args = _parse_args()
    registry = _registry()
    selected_polytopes = [registry[name] for name in args.polytopes]
    dataset = _build_dataset(selected_polytopes, transforms=args.transforms, seed=args.seed)

    sys.stdout.write("Profiling algorithms: " + ", ".join(args.algorithms) + "\n")
    sys.stdout.write("Polytopes: " + ", ".join(args.polytopes) + "\n")
    sys.stdout.write(f"Affine variants per polytope: {args.transforms}\n")
    sys.stdout.write(f"Total instances: {len(dataset)}\n")

    configs = (
        ProfileConfig(label=label, function=ALGORITHMS[label], repeats=args.repeats)
        for label in args.algorithms
    )
    for config in configs:
        _profile(config, dataset, top=args.top)


if __name__ == "__main__":
    main()
