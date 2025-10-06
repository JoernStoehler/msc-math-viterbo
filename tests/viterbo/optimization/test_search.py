"""Tests for the polytope search scaffolding."""

from __future__ import annotations

from viterbo.geometry.polytopes import catalog
from viterbo.optimization.search import enumerate_search_space


def test_enumerate_search_space_deterministic() -> None:
    first = enumerate_search_space(
        rng_seed=11,
        max_dimension=4,
        transforms_per_base=1,
        random_polytopes_per_dimension=1,
        max_candidates=12,
    )
    second = enumerate_search_space(
        rng_seed=11,
        max_dimension=4,
        transforms_per_base=1,
        random_polytopes_per_dimension=1,
        max_candidates=12,
    )
    assert len(first) == len(second)
    for poly_a, poly_b in zip(first, second, strict=True):
        assert poly_a.name == poly_b.name
        assert poly_a.B.shape == poly_b.B.shape
        assert poly_a.c.shape == poly_b.c.shape


def test_search_space_contains_catalog() -> None:
    search_space = enumerate_search_space(
        rng_seed=5,
        max_dimension=4,
        transforms_per_base=0,
        random_polytopes_per_dimension=0,
        max_candidates=32,
    )
    catalog_names = {poly.name for poly in catalog()}
    search_names = {poly.name for poly in search_space}
    assert catalog_names.issubset(search_names)
