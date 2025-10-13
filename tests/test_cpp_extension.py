import pytest
import torch

pytestmark = pytest.mark.smoke


def test_cpp_add_one_smoke():
    """Exercise C++ extension plumbing (falls back to Python if build fails)."""
    from viterbo._cpp import add_one

    x = torch.tensor([0.0, 1.0, 2.0])
    y = add_one(x)
    assert torch.allclose(y, x + 1)


def test_cpp_affine_scale_shift_smoke():
    """Ensure the multi-file affine example matches the Torch baseline."""
    from viterbo._cpp import affine_scale_shift

    x = torch.tensor([0.0, 0.5, 1.0])
    y = affine_scale_shift(x, scale=2.0, shift=-1.0)
    assert torch.allclose(y, x * 2.0 - 1.0)


def test_cpp_fallback_when_extension_missing(monkeypatch):
    """Force loaders to return None and verify Python fallbacks."""
    import viterbo._cpp as cpp

    cpp.clear_extension_caches()
    monkeypatch.setattr(cpp, "_load_add_one_extension", lambda: None)
    monkeypatch.setattr(cpp, "_load_affine_extension", lambda: None)

    x = torch.tensor([0.0, 1.0, 2.0])
    assert torch.allclose(cpp.add_one(x), x + 1)
    assert torch.allclose(cpp.affine_scale_shift(x, 2.0, -1.0), x * 2.0 - 1.0)
