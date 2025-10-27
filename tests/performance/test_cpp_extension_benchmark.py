import pytest
import torch

try:
    import pytest_benchmark  # noqa: F401
except ImportError:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="pytest-benchmark not installed")
else:
    pytestmark = [pytest.mark.smoke, pytest.mark.benchmark, pytest.mark.cpp]
    cpp = pytest.importorskip("viterbo._cpp")


def test_affine_scale_shift_cpp_benchmark(benchmark):
    """Benchmark the affine C++ path against the Torch baseline."""
    if not cpp.has_affine_extension():
        pytest.skip("affine C++ extension unavailable")

    torch.manual_seed(0)
    x = torch.randn(200_000, dtype=torch.float32)
    scale = 1.25
    shift = -0.5

    baseline = x * scale + shift

    result = benchmark(lambda: cpp.affine_scale_shift(x, scale, shift))
    torch.testing.assert_close(result, baseline)
