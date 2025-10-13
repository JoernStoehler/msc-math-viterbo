import torch
import pytest

try:
    import pytest_benchmark  # noqa: F401
except Exception:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="pytest-benchmark not installed")
else:
    pytestmark = [pytest.mark.smoke, pytest.mark.benchmark]


def test_support_benchmark(benchmark):
    """Benchmark the support function for a moderately sized point cloud."""
    from viterbo.math.geometry import support

    torch.manual_seed(0)
    pts = torch.randn(2000, 16)
    d = torch.randn(16)

    def run():
        return support(pts, d)

    s = benchmark(run)
    assert torch.is_tensor(s)
