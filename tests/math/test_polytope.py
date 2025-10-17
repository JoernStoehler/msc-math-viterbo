from __future__ import annotations

import math

import pytest
import torch

from tests.polytopes import STANDARD_POLYTOPES_BY_NAME
from viterbo.math.constructions import (
    matmul_vertices,
    rotated_regular_ngon2d,
    translate_vertices,
)
from viterbo.math.volume import volume

torch.set_default_dtype(torch.float64)


def test_matmul_and_translate_vertices() -> None:
    vertices = STANDARD_POLYTOPES_BY_NAME["right_triangle_area_one"].vertices
    matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    transformed = matmul_vertices(matrix, vertices)
    expected = vertices @ matrix.T
    torch.testing.assert_close(transformed, expected)
    translation = torch.tensor([1.0, -1.0])
    translated = translate_vertices(translation, transformed)
    expected_translated = expected + translation
    torch.testing.assert_close(translated, expected_translated)


def test_volume_known_shapes() -> None:
    square = STANDARD_POLYTOPES_BY_NAME["square_2d"].vertices
    unit_square = (square + 1.0) * 0.5
    assert math.isclose(volume(unit_square).item(), 1.0, rel_tol=1e-6, abs_tol=1e-6)

    cube = STANDARD_POLYTOPES_BY_NAME["hypercube_3d_unit"].vertices
    unit_cube = (cube + 1.0) * 0.5
    assert math.isclose(volume(unit_cube).item(), 1.0, rel_tol=1e-6, abs_tol=1e-6)

    hypercube_vertices = STANDARD_POLYTOPES_BY_NAME["hypercube_4d_unit"].vertices
    torch.testing.assert_close(
        volume(hypercube_vertices), torch.tensor(16.0, dtype=torch.get_default_dtype())
    )
    cross_poly_vertices = torch.cat([torch.eye(4), -torch.eye(4)], dim=0)
    expected_cross_poly_volume = torch.tensor(2.0 / 3.0, dtype=torch.get_default_dtype())
    torch.testing.assert_close(volume(cross_poly_vertices), expected_cross_poly_volume)


def test_rotated_regular_ngon() -> None:
    k = 6
    angle = math.pi / 6
    vertices, normals, offsets = rotated_regular_ngon2d(k, angle)
    assert vertices.shape == (k, 2)
    assert normals.shape == (k, 2)
    assert offsets.shape == (k,)
    # Check rotational symmetry: norms of offsets equal cos(pi/k)
    expected_offset = math.cos(math.pi / k)
    assert torch.allclose(offsets, torch.full_like(offsets, expected_offset), atol=1e-6)


@pytest.mark.skip(reason="higher-dimensional volume references not implemented")
def test_volume_against_lawrence_sign_decomposition() -> None:
    """Cross-check the torch implementation with a Lawrence-style reference solver.

    Once we introduce the Lawrence sign decomposition pipeline discussed in the
    ``volume`` docstring we will compare the torch result on a 4D zonotope
    against an external rational-arithmetic implementation (Büeler–Enge–Fukuda
    2000). This test documents the target verification workflow.
    """
    pytest.skip("higher-dimensional volume references not implemented")


@pytest.mark.skip(reason="triangulation backend not implemented")
def test_volume_via_triangulation_matches_simplex_formula() -> None:
    """Expect the triangulation backend to reproduce analytic simplex volumes.

    Once :func:`volume_via_triangulation` is implemented we will feed it the
    vertices of a 5D simplex with known volume ``1/120`` and assert exact
    agreement with the determinant-based formula described in the docstring.
    The comparison will use rational coordinates so the reference value is
    exact.
    """
    pytest.skip("triangulation backend not implemented")


@pytest.mark.skip(reason="Lawrence backend not implemented")
def test_volume_via_lawrence_certifies_cube() -> None:
    """Validate the sign decomposition against a halfspace description of a cube.

    The Lawrence routine should integrate the six supporting halfspaces of the
    unit cube and return volume ``1`` exactly, exposing the facet certificates
    so we can double-check the sign contributions match the analytic solution.
    """
    pytest.skip("Lawrence backend not implemented")


@pytest.mark.skip(reason="Monte Carlo backend not implemented")
def test_volume_via_monte_carlo_converges_with_rate() -> None:
    """Ensure the quasi-Monte Carlo estimator exhibits the expected ``O(1/N)`` rate.

    After implementing :func:`volume_via_monte_carlo` we will run the estimator on
    a 6D cross-polytope with known volume while increasing the ``samples`` count
    (powers of two Sobol points). The test will check that the empirical error
    halves whenever ``samples`` doubles, consistent with low-discrepancy
    sampling.
    """
    pytest.skip("Monte Carlo backend not implemented")
