"""Profiling helpers for the polytope EHZ capacity implementations."""

from __future__ import annotations

import cProfile
import pstats
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from viterbo import compute_ehz_capacity
from viterbo.ehz_fast import compute_ehz_capacity_fast


@dataclass
class ProfileConfig:
    """Configuration bundle describing a profiling experiment."""

    label: str
    function: Callable[[np.ndarray, np.ndarray], float]
    repeats: int


def _simplex_like_polytope_data(dimension: int = 4) -> tuple[np.ndarray, np.ndarray]:
    B = np.eye(dimension)
    extra = -np.ones((1, dimension))
    B = np.vstack((B, extra))
    c = np.ones(dimension + 1)
    c[-1] = dimension / 2
    return B, c


def _simplex_with_extra_facet_data() -> tuple[np.ndarray, np.ndarray]:
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
    return B, c


def _random_transformations(
    B: np.ndarray,
    c: np.ndarray,
    *,
    rng: np.random.Generator,
    count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate random linear transforms and translations of ``(B, c)``."""
    dimension = B.shape[1]
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(count):
        random_matrix = rng.normal(size=(dimension, dimension))
        q, _ = np.linalg.qr(random_matrix)
        scales = rng.uniform(0.7, 1.3, size=dimension)
        transform = q @ np.diag(scales)
        translated_B = B @ transform
        translation = rng.normal(scale=0.2, size=dimension)
        translated_c = c + translated_B @ translation
        results.append((translated_B, translated_c))
    return results


def _profile(config: ProfileConfig, dataset: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(config.repeats):
        for B, c in dataset:
            config.function(B, c)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    print(f"\n=== Profile: {config.label} ===")
    stats.print_stats(15)


def main() -> None:
    """Profile the reference and optimized capacity implementations."""
    rng = np.random.default_rng(2024)
    base_pairs = (
        _simplex_like_polytope_data(),
        _simplex_with_extra_facet_data(),
    )
    dataset: list[tuple[np.ndarray, np.ndarray]] = []
    for B, c in base_pairs:
        dataset.append((B, c))
        dataset.extend(_random_transformations(B, c, rng=rng, count=15))

    configs = (
        ProfileConfig(label="reference", function=compute_ehz_capacity, repeats=5),
        ProfileConfig(label="fast", function=compute_ehz_capacity_fast, repeats=5),
    )

    for config in configs:
        _profile(config, dataset)


if __name__ == "__main__":
    main()
