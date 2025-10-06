"""Profiling helpers for the polytope EHZ capacity implementations."""

from __future__ import annotations

import argparse
import cProfile
import pstats
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import jax
import numpy as np

from viterbo.geometry.polytopes import Polytope, catalog, random_transformations
from viterbo.symplectic.capacity import compute_ehz_capacity_reference
from viterbo.symplectic.capacity.facet_normals.fast import compute_ehz_capacity_fast

Algorithm = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class ProfileConfig:
    """Configuration bundle describing a profiling experiment."""

    label: str
    function: Algorithm
    repeats: int


ALGORITHMS: dict[str, Algorithm] = {
    "reference": compute_ehz_capacity_reference,
    "fast": compute_ehz_capacity_fast,
}


def _registry() -> dict[str, Polytope]:
    return {poly.name: poly for poly in catalog()}


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
    polytopes: Iterable[Polytope],
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
        B, c = poly.halfspace_data()
        dataset.append((poly.name, B, c))
        if transforms:
            variants = random_transformations(poly, key=key, count=transforms)
            for index, variant in enumerate(variants):
                variant_B, variant_c = variant.halfspace_data()
                label = f"{poly.name}-variant-{index}"
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
    print(f"\n=== Profile: {config.label} ===")
    stats.print_stats(top)


def main() -> None:
    """Profile the reference and optimized capacity implementations."""
    args = _parse_args()
    registry = _registry()
    selected_polytopes = [registry[name] for name in args.polytopes]
    dataset = _build_dataset(selected_polytopes, transforms=args.transforms, seed=args.seed)

    print("Profiling algorithms:", ", ".join(args.algorithms))
    print("Polytopes:", ", ".join(args.polytopes))
    print(f"Affine variants per polytope: {args.transforms}")
    print(f"Total instances: {len(dataset)}")

    configs = (
        ProfileConfig(label=label, function=ALGORITHMS[label], repeats=args.repeats)
        for label in args.algorithms
    )
    for config in configs:
        _profile(config, dataset, top=args.top)


if __name__ == "__main__":
    main()
