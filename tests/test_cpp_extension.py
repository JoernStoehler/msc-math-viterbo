import pytest
import torch

pytestmark = pytest.mark.smoke


def test_cpp_add_one_smoke():
    """Exercise C++ extension plumbing (falls back to Python if build fails)."""
    from viterbo._cpp import add_one

    x = torch.tensor([0.0, 1.0, 2.0])
    y = add_one(x)
    assert torch.allclose(y, x + 1)
