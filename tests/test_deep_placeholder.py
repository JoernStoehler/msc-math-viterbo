import pytest


@pytest.mark.deep
def test_deep_placeholder():
    """Deep-tier placeholder to verify marker wiring and CI selection."""
    assert 1 + 1 == 2

