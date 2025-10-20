from __future__ import annotations

import torch

from viterbo.math.symplectic import random_symplectic_matrix, symplectic_form


torch.set_default_dtype(torch.float64)


def test_random_symplectic_preserves_J() -> None:
    for d in (4, 6):
        J = symplectic_form(d)
        M = random_symplectic_matrix(d, seed=0)
        left = M.T @ J @ M
        torch.testing.assert_close(left, J, atol=1e-10, rtol=0.0)

