"""Tests for minimal-action Minkowski billiards in planar Lagrangian products."""

from __future__ import annotations

import math

import torch

from viterbo.math.constructions import rotated_regular_ngon2d
from viterbo.math.minimal_action import minimal_action_cycle_lagrangian_product
from viterbo.math.polytope import vertices_to_halfspaces

torch.set_default_dtype(torch.float64)


def test_pentagon_pair_matches_known_constant() -> None:
    vertices_q, _, _ = rotated_regular_ngon2d(5, 0.0)
    vertices_p, normals_p, offsets_p = rotated_regular_ngon2d(5, -math.pi / 2)

    capacity_general, cycle_general = minimal_action_cycle_lagrangian_product(
        vertices_q, normals_p, offsets_p
    )
    capacity_two, cycle_two = minimal_action_cycle_lagrangian_product(
        vertices_q, normals_p, offsets_p, max_bounces=2
    )

    expected = 2.0 * math.cos(math.pi / 10.0) * (1.0 + math.cos(math.pi / 5.0))
    torch.testing.assert_close(capacity_general, torch.tensor(expected), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(capacity_two, capacity_general, rtol=1e-12, atol=1e-12)
    assert cycle_general.size(0) == 5
    assert cycle_two.size(0) == 5


def test_three_bounce_orbit_beats_two_bounce_candidate() -> None:
    vertices_q = torch.tensor(
        [
            [-1.70172287, -1.26811867],
            [-1.56413513, -1.89508182],
            [1.19150140, -0.84596686],
            [1.06428033, 0.66886455],
            [-1.40051732, -0.49548974],
        ]
    )
    vertices_p = torch.tensor(
        [
            [-1.92487754, -0.41382604],
            [1.65055805, 0.15091143],
            [0.53522767, 0.94426645],
            [1.30412803, 1.78716638],
            [-1.37802231, 1.88839061],
        ]
    )
    normals_p, offsets_p = vertices_to_halfspaces(vertices_p)

    capacity_general, cycle_general = minimal_action_cycle_lagrangian_product(
        vertices_q, normals_p, offsets_p
    )
    capacity_two, cycle_two = minimal_action_cycle_lagrangian_product(
        vertices_q, normals_p, offsets_p, max_bounces=2
    )

    assert cycle_general.size(0) == 7
    assert cycle_two.size(0) == 5
    assert capacity_general < capacity_two


def test_translation_and_permutation_invariance() -> None:
    vertices_q = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.2],
            [1.5, 1.5],
            [-0.5, 1.3],
            [-1.2, 0.2],
        ]
    )
    vertices_p = torch.tensor(
        [
            [1.2, 0.0],
            [0.3, 1.5],
            [-1.1, 0.7],
            [-0.6, -0.9],
            [1.1, -0.4],
        ]
    )
    normals_p, offsets_p = vertices_to_halfspaces(vertices_p)

    capacity_base, cycle_base = minimal_action_cycle_lagrangian_product(
        vertices_q, normals_p, offsets_p
    )

    permuted_vertices_q = vertices_q[[2, 4, 0, 1, 3]]
    capacity_perm, cycle_perm = minimal_action_cycle_lagrangian_product(
        permuted_vertices_q, normals_p, offsets_p
    )

    translation = torch.tensor([3.1, -2.4])
    translated_vertices_q = vertices_q + translation
    capacity_trans, cycle_trans = minimal_action_cycle_lagrangian_product(
        translated_vertices_q, normals_p, offsets_p
    )

    torch.testing.assert_close(capacity_perm, capacity_base, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(capacity_trans, capacity_base, rtol=1e-8, atol=1e-8)

    torch.testing.assert_close(
        cycle_trans[:, :2], cycle_base[:, :2] + translation, rtol=1e-8, atol=1e-8
    )
    assert cycle_perm.size(0) == cycle_base.size(0)
    assert cycle_trans.size(0) == cycle_base.size(0)
