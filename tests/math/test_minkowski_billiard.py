"""Tests for minimal-action Minkowski billiards in planar Lagrangian products."""

from __future__ import annotations

import math

import torch
from tests.polytopes import PLANAR_POLYTOPE_PAIRS

from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product

torch.set_default_dtype(torch.float64)


def test_pentagon_pair_matches_known_constant() -> None:
    pentagon_q, pentagon_p = PLANAR_POLYTOPE_PAIRS["pentagon_product"]

    capacity_general, cycle_general = minimal_action_cycle_lagrangian_product(
        pentagon_q.vertices, pentagon_p.normals, pentagon_p.offsets
    )
    capacity_two, cycle_two = minimal_action_cycle_lagrangian_product(
        pentagon_q.vertices, pentagon_p.normals, pentagon_p.offsets, max_bounces=2
    )

    expected = 2.0 * math.cos(math.pi / 10.0) * (1.0 + math.cos(math.pi / 5.0))
    torch.testing.assert_close(capacity_general, torch.tensor(expected), rtol=0.0, atol=1e-7)
    torch.testing.assert_close(capacity_two, capacity_general, rtol=1e-12, atol=1e-12)
    assert cycle_general.size(0) == pentagon_q.vertices.size(0)
    assert cycle_two.size(0) == pentagon_q.vertices.size(0)


def test_three_bounce_orbit_beats_two_bounce_candidate() -> None:
    poly_q, poly_p = PLANAR_POLYTOPE_PAIRS["minkowski_three_bounce"]

    capacity_general, cycle_general = minimal_action_cycle_lagrangian_product(
        poly_q.vertices, poly_p.normals, poly_p.offsets
    )
    capacity_two, cycle_two = minimal_action_cycle_lagrangian_product(
        poly_q.vertices, poly_p.normals, poly_p.offsets, max_bounces=2
    )

    assert cycle_general.size(0) == 7
    assert cycle_two.size(0) == 5
    assert capacity_general < capacity_two


def test_translation_and_permutation_invariance() -> None:
    poly_q, poly_p = PLANAR_POLYTOPE_PAIRS["minkowski_invariance"]

    capacity_base, cycle_base = minimal_action_cycle_lagrangian_product(
        poly_q.vertices, poly_p.normals, poly_p.offsets
    )

    permuted_vertices_q = poly_q.vertices[[2, 4, 0, 1, 3]]
    capacity_perm, cycle_perm = minimal_action_cycle_lagrangian_product(
        permuted_vertices_q, poly_p.normals, poly_p.offsets
    )

    translation = torch.tensor([3.1, -2.4])
    translated_vertices_q = poly_q.vertices + translation
    capacity_trans, cycle_trans = minimal_action_cycle_lagrangian_product(
        translated_vertices_q, poly_p.normals, poly_p.offsets
    )

    torch.testing.assert_close(capacity_perm, capacity_base, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(capacity_trans, capacity_base, rtol=1e-8, atol=1e-8)

    torch.testing.assert_close(
        cycle_trans[:, :2], cycle_base[:, :2] + translation, rtol=1e-8, atol=1e-8
    )
    assert cycle_perm.size(0) == cycle_base.size(0)
    assert cycle_trans.size(0) == cycle_base.size(0)


def test_noisy_pentagon_pair_smoke() -> None:
    # Smoke test on the non-regular pentagon pair to ensure the solver
    # handles symmetry-broken inputs robustly.
    pent_q, pent_p = PLANAR_POLYTOPE_PAIRS["noisy_pentagon_product"]

    capacity_general, cycle_general = minimal_action_cycle_lagrangian_product(
        pent_q.vertices, pent_p.normals, pent_p.offsets
    )
    capacity_two, cycle_two = minimal_action_cycle_lagrangian_product(
        pent_q.vertices, pent_p.normals, pent_p.offsets, max_bounces=2
    )

    assert capacity_general.ndim == 0 and capacity_general.item() > 0.0
    assert capacity_two.ndim == 0 and capacity_two.item() > 0.0
    assert cycle_general.ndim == 2 and cycle_general.size(1) == 4
    assert cycle_two.ndim == 2 and cycle_two.size(1) == 4
    # Both cycles are closed polylines in R^4.
    torch.testing.assert_close(cycle_general[0], cycle_general[-1], atol=1e-9, rtol=0.0)
    torch.testing.assert_close(cycle_two[0], cycle_two[-1], atol=1e-9, rtol=0.0)
