"""Stubbed polytope generators for the modern API."""

from __future__ import annotations

from typing import Iterable

from jaxtyping import Array, Float

from .types import GeneratorMetadata, PolytopeBundle


def sample_uniform_ball(
    key: Array,
    dimension: int,
    *,
    num_samples: int,
) -> Iterable[tuple[PolytopeBundle, GeneratorMetadata]]:
    """Yield polytopes sampled from the unit ball."""

    raise NotImplementedError


def sample_product(
    generator_a: Iterable[tuple[PolytopeBundle, GeneratorMetadata]],
    generator_b: Iterable[tuple[PolytopeBundle, GeneratorMetadata]],
) -> Iterable[tuple[PolytopeBundle, GeneratorMetadata]]:
    """Combine two generator streams via Cartesian product."""

    raise NotImplementedError


def enumerate_lattice(
    dimension: int,
    bounds: Float[Array, " dimension 2"],
) -> Iterable[tuple[PolytopeBundle, GeneratorMetadata]]:
    """Enumerate lattice polytopes within the provided bounds."""

    raise NotImplementedError
