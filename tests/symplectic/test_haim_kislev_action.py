"""Unit tests for the Haim–Kislev action helper."""

from __future__ import annotations

import math

import pytest

from viterbo.geometry.polytopes import haim_kislev_action, truncated_simplex_four_dim


def test_haim_kislev_action_valid_order_matches_reference_capacity() -> None:
    polytope = truncated_simplex_four_dim()
    subset = (0, 1, 2, 3, 4)
    order = (2, 0, 4, 3, 1)
    action = haim_kislev_action(polytope.B, polytope.c, subset=subset, order=order)
    reference_capacity = polytope.reference_capacity
    assert reference_capacity is not None
    assert math.isclose(action, reference_capacity)


def test_haim_kislev_action_invalid_order_raises_value_error() -> None:
    polytope = truncated_simplex_four_dim()
    subset = (0, 1, 2, 3, 4)
    invalid_order = (2, 0, 4, 4, 1)
    with pytest.raises(ValueError):
        haim_kislev_action(polytope.B, polytope.c, subset=subset, order=invalid_order)
