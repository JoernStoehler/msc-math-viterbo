from __future__ import annotations

import torch

from viterbo.math.symplectic import random_symplectic_matrix, symplectic_form

torch.set_default_dtype(torch.float64)


def test_symplectic_form_structure() -> None:
    j = symplectic_form(4)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(j, expected)


def test_random_symplectic_matrix_preserves_form() -> None:
    dimension = 4
    seed = 42
    matrix = random_symplectic_matrix(dimension, seed)
    j = symplectic_form(dimension)
    lhs = matrix.T @ j @ matrix
    torch.testing.assert_close(lhs, j, atol=1e-6, rtol=1e-6)
