from __future__ import annotations

import math

import torch

from viterbo.math.capacity_ehz.algorithms import capacity_ehz_algorithm2
from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product
from viterbo.math.capacity_ehz.stubs import oriented_edge_spectrum_4d
from viterbo.math.constructions import (
    lagrangian_product,
    random_polytope_algorithm2,
    rotated_regular_ngon2d,
)
from viterbo.math.polytope import vertices_to_halfspaces

torch.set_default_dtype(torch.float64)


def test_oriented_edge_matches_lagrangian_product_solver() -> None:
    vertices_q, _, _ = rotated_regular_ngon2d(5, 0.0)
    vertices_p, normals_p, offsets_p = rotated_regular_ngon2d(5, -math.pi / 2)
    vertices, normals, offsets = lagrangian_product(vertices_q, vertices_p)
    capacity_edge = oriented_edge_spectrum_4d(vertices, normals, offsets)
    capacity_lp, _ = minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p)
    torch.testing.assert_close(capacity_edge, capacity_lp, atol=1e-8, rtol=1e-8)


def test_capacity_algorithm2_falls_back_to_oriented_edge() -> None:
    seed = 11
    vertices, normals, offsets = random_polytope_algorithm2(seed, num_vertices=18, dimension=4)
    capacity_direct = oriented_edge_spectrum_4d(vertices, normals, offsets)
    capacity_algo = capacity_ehz_algorithm2(vertices)
    torch.testing.assert_close(capacity_direct, capacity_algo, atol=1e-7, rtol=1e-7)


def test_oriented_edge_translation_and_scaling_invariance() -> None:
    seed = 23
    vertices, normals, offsets = random_polytope_algorithm2(seed, num_vertices=20, dimension=4)
    capacity_base = oriented_edge_spectrum_4d(vertices, normals, offsets)

    translation = torch.tensor([0.3, -1.1, 0.2, 0.9], dtype=torch.get_default_dtype())
    translated_vertices = vertices + translation
    normals_trans, offsets_trans = vertices_to_halfspaces(translated_vertices)
    capacity_trans = oriented_edge_spectrum_4d(translated_vertices, normals_trans, offsets_trans)
    torch.testing.assert_close(capacity_trans, capacity_base, atol=1e-7, rtol=1e-7)

    scale = 1.7
    scaled_vertices = vertices * scale
    normals_scaled, offsets_scaled = vertices_to_halfspaces(scaled_vertices)
    capacity_scaled = oriented_edge_spectrum_4d(scaled_vertices, normals_scaled, offsets_scaled)
    torch.testing.assert_close(capacity_scaled, capacity_base * (scale**2), atol=1e-6, rtol=1e-6)
