"""Tests for the polytope search scaffolding."""

from __future__ import annotations

import pytest

from viterbo.geometry.polytopes import catalog
from viterbo.optimization.search import enumerate_search_space, iter_search_space


@pytest.mark.goal_code
def test_enumerate_search_space_deterministic() -> None:
    """Enumerating the search space with identical seeds yields identical sequences."""
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


@pytest.mark.goal_math
def test_search_space_contains_catalog() -> None:
    """The search enumeration includes every canonical polytope from the catalog."""
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


@pytest.mark.goal_code
def test_iter_search_space_respects_max_candidates() -> None:
    """The iterator stops once it yields the requested number of candidates."""
    polytopes = list(
        iter_search_space(
            max_candidates=5,
            transforms_per_base=0,
            random_polytopes_per_dimension=0,
            max_dimension=3,
        )
    )
    assert len(polytopes) == 5


@pytest.mark.goal_code
def test_iter_search_space_rejects_unknown_kwargs() -> None:
    """`iter_search_space` raises when callers supply unsupported keyword arguments."""
    with pytest.raises(TypeError, match="Unknown keyword arguments"):
        next(iter_search_space(unknown=1))


@pytest.mark.goal_code
def test_iter_search_space_honours_dimension_cap() -> None:
    """Generated polytopes do not exceed the requested maximum dimension."""
    polytopes = list(
        iter_search_space(
            max_dimension=3,
            transforms_per_base=0,
            random_polytopes_per_dimension=0,
            max_candidates=20,
        )
    )
    assert not any("gonx" in poly.name for poly in polytopes)
