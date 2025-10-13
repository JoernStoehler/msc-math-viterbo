"""Smoke tests for symplectic invariants and capacities.

The tests in this module target high-level mathematical behaviour that the
placeholder implementations in :mod:`viterbo.math.symplectic` must satisfy once
completed.  They focus on invariance properties, scaling laws, and benchmark
values extracted from the literature (notably the 2024 counterexample to
Viterbo's conjecture recorded in the thesis notes).
"""

from __future__ import annotations

import math

import pytest
import torch

from viterbo.math.geometry import matmul_vertices, translate_vertices, volume
from viterbo.math.halfspaces import vertices_to_halfspaces
from viterbo.math.symplectic import (
    capacity_ehz_algorithm1,
    capacity_ehz_algorithm2,
    minimal_action_cycle,
    random_symplectic_matrix,
    systolic_ratio,
)


torch.set_default_dtype(torch.float64)


def _call_or_skip(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{func.__name__} is not implemented yet")


def _unit_square_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )


def _square_geometry():
    vertices = _unit_square_vertices()
    normals, offsets = vertices_to_halfspaces(vertices)
    return vertices, normals, offsets


def test_volume_translation_invariance() -> None:
    vertices = torch.tensor([[-1.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    translated = translate_vertices(torch.tensor([3.5, -1.25]), vertices)
    original_volume = volume(vertices)
    translated_volume = volume(translated)
    torch.testing.assert_close(translated_volume, original_volume)


def test_volume_scaling_homogeneity() -> None:
    vertices = _unit_square_vertices()
    scale = 1.75
    scaled_vertices = vertices * scale
    scaled_volume = volume(scaled_vertices)
    base_volume = volume(vertices)
    dimension = vertices.size(1)
    torch.testing.assert_close(scaled_volume, base_volume * scale**dimension, atol=1e-10, rtol=1e-10)


def test_volume_known_cube_value() -> None:
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    cube_volume = volume(vertices)
    torch.testing.assert_close(cube_volume, torch.tensor(8.0))


def test_capacity_algorithms_agree_on_square() -> None:
    vertices, normals, offsets = _square_geometry()
    capacity_h = _call_or_skip(capacity_ehz_algorithm1, normals, offsets)
    capacity_v = _call_or_skip(capacity_ehz_algorithm2, vertices)
    torch.testing.assert_close(capacity_h, capacity_v)


def test_capacity_invariant_under_symplectic_map() -> None:
    vertices, normals, offsets = _square_geometry()
    matrix = random_symplectic_matrix(2, seed=1234)
    transformed_vertices = matmul_vertices(matrix, vertices)
    inverse_matrix = torch.linalg.inv(matrix)
    transformed_normals = normals @ inverse_matrix
    capacity_original = _call_or_skip(capacity_ehz_algorithm1, normals, offsets)
    capacity_transformed = _call_or_skip(capacity_ehz_algorithm1, transformed_normals, offsets)
    torch.testing.assert_close(capacity_transformed, capacity_original, rtol=1e-6, atol=1e-6)
    capacity_v_original = _call_or_skip(capacity_ehz_algorithm2, vertices)
    capacity_v_transformed = _call_or_skip(capacity_ehz_algorithm2, transformed_vertices)
    torch.testing.assert_close(capacity_v_transformed, capacity_v_original, rtol=1e-6, atol=1e-6)


def test_capacity_translation_invariance() -> None:
    vertices, normals, offsets = _square_geometry()
    translation = torch.tensor([0.6, -1.2])
    translated_vertices = translate_vertices(translation, vertices)
    translated_offsets = offsets + normals @ translation
    capacity_original = _call_or_skip(capacity_ehz_algorithm1, normals, offsets)
    capacity_translated = _call_or_skip(capacity_ehz_algorithm1, normals, translated_offsets)
    torch.testing.assert_close(capacity_translated, capacity_original, rtol=1e-6, atol=1e-6)
    capacity_v_original = _call_or_skip(capacity_ehz_algorithm2, vertices)
    capacity_v_translated = _call_or_skip(capacity_ehz_algorithm2, translated_vertices)
    torch.testing.assert_close(capacity_v_translated, capacity_v_original, rtol=1e-6, atol=1e-6)


def test_capacity_scaling_conformality() -> None:
    vertices, _, _ = _square_geometry()
    normals, offsets = vertices_to_halfspaces(vertices)
    scale = 1.3
    scaled_vertices = vertices * scale
    scaled_normals, scaled_offsets = vertices_to_halfspaces(scaled_vertices)
    capacity_original = _call_or_skip(capacity_ehz_algorithm1, normals, offsets)
    capacity_scaled = _call_or_skip(capacity_ehz_algorithm1, scaled_normals, scaled_offsets)
    torch.testing.assert_close(capacity_scaled, capacity_original * scale**2, rtol=1e-6, atol=1e-6)
    capacity_v_original = _call_or_skip(capacity_ehz_algorithm2, vertices)
    capacity_v_scaled = _call_or_skip(capacity_ehz_algorithm2, scaled_vertices)
    torch.testing.assert_close(capacity_v_scaled, capacity_v_original * scale**2, rtol=1e-6, atol=1e-6)


def test_minimal_action_cycle_matches_capacity() -> None:
    vertices, normals, offsets = _square_geometry()
    capacity_from_cycle, cycle = _call_or_skip(minimal_action_cycle, vertices, normals, offsets)
    capacity_from_halfspaces = _call_or_skip(capacity_ehz_algorithm1, normals, offsets)
    torch.testing.assert_close(capacity_from_cycle, capacity_from_halfspaces, rtol=1e-6, atol=1e-6)
    assert cycle.ndim == 2 and cycle.size(1) == vertices.size(1)


def test_minimal_action_cycle_translation_covariance() -> None:
    vertices, normals, offsets = _square_geometry()
    translation = torch.tensor([-0.3, 2.7])
    translated_vertices = translate_vertices(translation, vertices)
    translated_offsets = offsets + normals @ translation
    capacity_original, cycle = _call_or_skip(minimal_action_cycle, vertices, normals, offsets)
    capacity_translated, cycle_translated = _call_or_skip(
        minimal_action_cycle,
        translated_vertices,
        normals,
        translated_offsets,
    )
    torch.testing.assert_close(capacity_translated, capacity_original, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cycle_translated, cycle + translation, rtol=1e-6, atol=1e-6)


def test_minimal_action_cycle_symplectic_covariance() -> None:
    vertices, normals, offsets = _square_geometry()
    matrix = random_symplectic_matrix(2, seed=4321)
    transformed_vertices = matmul_vertices(matrix, vertices)
    inverse_matrix = torch.linalg.inv(matrix)
    transformed_normals = normals @ inverse_matrix
    capacity_original, cycle = _call_or_skip(minimal_action_cycle, vertices, normals, offsets)
    capacity_transformed, cycle_transformed = _call_or_skip(
        minimal_action_cycle,
        transformed_vertices,
        transformed_normals,
        offsets,
    )
    torch.testing.assert_close(capacity_transformed, capacity_original, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cycle_transformed, cycle @ matrix.T, rtol=1e-6, atol=1e-6)


def test_systolic_ratio_matches_counterexample_constant() -> None:
    volume_value = torch.tensor(1.0)
    capacity_value = torch.tensor(math.sqrt(2.0 * (math.sqrt(5.0) + 3.0) / 5.0))
    ratio = _call_or_skip(systolic_ratio, volume_value, capacity_value)
    expected = torch.tensor(5.0 / (2.0 * (math.sqrt(5.0) + 3.0)))
    torch.testing.assert_close(ratio, expected, rtol=1e-9, atol=1e-9)
