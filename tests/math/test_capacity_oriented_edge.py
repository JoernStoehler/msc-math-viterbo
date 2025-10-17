from __future__ import annotations

import math

import pytest
import torch

from tests.polytopes import PLANAR_POLYTOPE_PAIRS
from viterbo.math.capacity_ehz.algorithms import capacity_ehz_algorithm2
from viterbo.math.capacity_ehz.lagrangian_product import minimal_action_cycle_lagrangian_product
from viterbo.math.capacity_ehz.stubs import oriented_edge_spectrum_4d
from viterbo.math.constructions import lagrangian_product, matmul_vertices
from viterbo.math.polytope import vertices_to_halfspaces

torch.set_default_dtype(torch.float64)


def test_oriented_edge_matches_lagrangian_product_solver() -> None:
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    vertices, normals, offsets = lagrangian_product(square_q.vertices, square_p.vertices)
    capacity_edge = oriented_edge_spectrum_4d(vertices, normals, offsets)
    capacity_lp, _ = minimal_action_cycle_lagrangian_product(
        square_q.vertices, square_p.normals, square_p.offsets
    )
    torch.testing.assert_close(capacity_edge, capacity_lp, atol=1e-8, rtol=1e-8)


def test_capacity_algorithm2_falls_back_to_oriented_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    base_vertices, base_normals, base_offsets = lagrangian_product(
        square_q.vertices, square_p.vertices
    )
    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.0, 1.1, -0.3, 0.0],
            [0.1, 0.0, 1.0, 0.4],
            [0.0, -0.2, 0.0, 1.2],
        ],
        dtype=torch.get_default_dtype(),
    )
    vertices = matmul_vertices(matrix, base_vertices)
    sentinel = torch.tensor(3.14159, dtype=vertices.dtype)
    called: dict[str, bool] = {"flag": False}

    def fake_oriented(edge_vertices: torch.Tensor, edge_normals: torch.Tensor, edge_offsets: torch.Tensor) -> torch.Tensor:
        called["flag"] = True
        assert edge_vertices.shape[1] == 4
        assert edge_normals.shape[1] == 4
        assert edge_offsets.ndim == 1
        return sentinel

    monkeypatch.setattr(
        "viterbo.math.capacity_ehz.algorithms.oriented_edge_spectrum_4d",
        fake_oriented,
    )

    capacity_algo = capacity_ehz_algorithm2(vertices)
    assert called["flag"] is True
    torch.testing.assert_close(capacity_algo, sentinel, atol=1e-9, rtol=0.0)


def test_oriented_edge_translation_and_scaling_invariance() -> None:
    square_q, square_p = PLANAR_POLYTOPE_PAIRS["square_product"]
    vertices, normals, offsets = lagrangian_product(square_q.vertices, square_p.vertices)
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
