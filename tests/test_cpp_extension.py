import pytest
import torch

pytestmark = [pytest.mark.smoke, pytest.mark.cpp]

cpp = pytest.importorskip("viterbo._cpp")


def test_cpp_add_one_smoke():
    """Exercise the compiled add_one op and compare to a Torch baseline."""
    x = torch.tensor([0.0, 1.0, 2.0])
    y = cpp.add_one(x)
    assert torch.allclose(y, x + 1)


def test_cpp_affine_scale_shift_smoke():
    """Ensure the multi-file affine example matches the Torch baseline."""
    x = torch.tensor([0.0, 0.5, 1.0])
    y = cpp.affine_scale_shift(x, scale=2.0, shift=-1.0)
    assert torch.allclose(y, x * 2.0 - 1.0)
