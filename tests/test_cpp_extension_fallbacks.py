import importlib
from types import SimpleNamespace

import pytest
import torch


pytestmark = [pytest.mark.smoke]


def _fake_ext_module():
    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    def affine_scale_shift(x: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
        return x * scale + shift

    return SimpleNamespace(add_one=add_one, affine_scale_shift=affine_scale_shift)


def test_extension_helpers_fail_on_build_error(monkeypatch):
    # First import with a fake extension to avoid any actual compilation.
    # Ensure we exercise the CC→CXX propagation branch in the loader.
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("CC", "cc")

    monkeypatch.setattr(
        "torch.utils.cpp_extension.load", lambda *a, **k: _fake_ext_module(), raising=True
    )
    cpp = importlib.import_module("viterbo._cpp")

    # Sanity: the happy path works with the fake module.
    x = torch.tensor([0.0, 1.0])
    assert torch.allclose(cpp.add_one(x), x + 1)
    assert torch.allclose(cpp.affine_scale_shift(x, 2.0, -1.0), x * 2.0 - 1.0)

    # Happy path availability helpers are True with the fake module.
    assert cpp.has_add_one_extension() is True
    assert cpp.has_affine_extension() is True

    # Now simulate build/import failure by making torch's loader raise ImportError.
    def _boom(*_a, **_k):
        raise ImportError("simulated build failure")

    monkeypatch.setattr("torch.utils.cpp_extension.load", _boom, raising=True)
    # Also patch the symbol imported into the module under test.
    monkeypatch.setattr("viterbo._cpp.load", _boom, raising=True)

    # Clearing caches forces the lazy loaders to call into the failing path.
    cpp.clear_extension_caches()

    # Availability helpers report False on failure.
    assert cpp.has_add_one_extension() is False
    assert cpp.has_affine_extension() is False

    # Functional helpers raise ImportError with the standardized diagnostics.
    with pytest.raises(ImportError) as ei1:
        cpp.add_one(x)
    assert "Failed to build or load" in str(ei1.value)
    assert "viterbo_add_one_ext" in str(ei1.value)

    with pytest.raises(ImportError) as ei2:
        cpp.affine_scale_shift(x, 2.0, -1.0)
    assert "Failed to build or load" in str(ei2.value)
    assert "viterbo_affine_ext" in str(ei2.value)

    # Clearing again keeps behavior stable/predictable.
    cpp.clear_extension_caches()
    assert cpp.has_add_one_extension() is False
    with pytest.raises(ImportError):
        cpp.add_one(x)
